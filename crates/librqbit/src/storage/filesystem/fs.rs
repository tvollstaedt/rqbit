use std::{
    fs::OpenOptions,
    io::IoSlice,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::Context;
use tracing::warn;

use crate::{
    storage::{StorageFactoryExt, filesystem::opened_file::OurFileExt},
    torrent_state::{ManagedTorrentShared, TorrentMetadata},
};

use crate::storage::{StorageFactory, TorrentStorage};

use super::opened_file::{FilePool, OpenedFile};

#[derive(Default, Clone, Copy)]
pub struct FilesystemStorageFactory {}

impl StorageFactory for FilesystemStorageFactory {
    type Storage = FilesystemStorage;

    fn create(
        &self,
        shared: &ManagedTorrentShared,
        _metadata: &TorrentMetadata,
    ) -> anyhow::Result<FilesystemStorage> {
        Ok(FilesystemStorage {
            output_folder: shared.options.output_folder.clone(),
            opened_files: Default::default(),
            pool: FilePool::global(),
        })
    }

    fn clone_box(&self) -> crate::storage::BoxStorageFactory {
        self.boxed()
    }
}

/// One entry per torrent file; handles are opened on first access and the
/// process-wide pool keeps at most its cap open (oldest closed first).
pub struct FilesystemStorage {
    pub(crate) output_folder: PathBuf,
    pub(crate) opened_files: Vec<Arc<OpenedFile>>,
    pub(crate) pool: Arc<FilePool>,
}

impl FilesystemStorage {
    #[allow(dead_code)]
    pub(crate) fn take_fs(&self) -> anyhow::Result<Self> {
        Ok(Self {
            opened_files: self
                .opened_files
                .iter()
                .map(|f| f.take_clone().map(Arc::new))
                .collect::<anyhow::Result<Vec<_>>>()?,
            output_folder: self.output_folder.clone(),
            pool: self.pool.clone(),
        })
    }

    fn file(&self, file_id: usize) -> anyhow::Result<&Arc<OpenedFile>> {
        self.opened_files.get(file_id).context("no such file")
    }
}

impl TorrentStorage for FilesystemStorage {
    fn pread_exact(&self, file_id: usize, offset: u64, buf: &mut [u8]) -> anyhow::Result<()> {
        self.file(file_id)?.lock_read()?.pread_exact(offset, buf)
    }

    fn pwrite_all(&self, file_id: usize, offset: u64, buf: &[u8]) -> anyhow::Result<()> {
        let of = self.file(file_id)?;
        #[cfg(windows)]
        return of.try_mark_sparse()?.pwrite_all(offset, buf);
        #[cfg(not(windows))]
        return of.lock_read()?.pwrite_all(offset, buf);
    }

    fn pwrite_all_vectored(
        &self,
        file_id: usize,
        offset: u64,
        bufs: [IoSlice<'_>; 2],
    ) -> anyhow::Result<usize> {
        let of = self.file(file_id)?;
        #[cfg(windows)]
        return of.try_mark_sparse()?.pwrite_all_vectored(offset, bufs);
        #[cfg(not(windows))]
        return of.lock_read()?.pwrite_all_vectored(offset, bufs);
    }

    fn remove_file(&self, _file_id: usize, filename: &Path) -> anyhow::Result<()> {
        Ok(std::fs::remove_file(self.output_folder.join(filename))?)
    }

    fn ensure_file_length(&self, file_id: usize, len: u64) -> anyhow::Result<()> {
        let f = self.file(file_id)?;
        #[cfg(windows)]
        f.try_mark_sparse()?;
        Ok(f.lock_read()?.set_len(len)?)
    }

    fn take(&self) -> anyhow::Result<Box<dyn TorrentStorage>> {
        Ok(Box::new(self.take_fs()?))
    }

    fn remove_directory_if_empty(&self, path: &Path) -> anyhow::Result<()> {
        let path = self.output_folder.join(path);
        if !path.is_dir() {
            anyhow::bail!("cannot remove dir: {path:?} is not a directory")
        }
        if std::fs::read_dir(&path)?.count() == 0 {
            std::fs::remove_dir(&path).with_context(|| format!("error removing {path:?}"))
        } else {
            warn!("did not remove {path:?} as it was not empty");
            Ok(())
        }
    }

    /// Creates every file (so placeholders exist and `allow_overwrite =
    /// false` is honoured) but holds no handle: the first access opens it.
    fn init(
        &mut self,
        shared: &ManagedTorrentShared,
        metadata: &TorrentMetadata,
    ) -> anyhow::Result<()> {
        let mut files = Vec::<Arc<OpenedFile>>::new();
        for file_details in metadata.file_infos.iter() {
            let mut full_path = self.output_folder.clone();
            let relative_path = &file_details.relative_filename;
            full_path.push(relative_path);

            if file_details.attrs.padding {
                files.push(Arc::new(OpenedFile::new_dummy(self.pool.clone())));
                continue;
            };
            std::fs::create_dir_all(full_path.parent().context("bug: no parent")?)?;
            if shared.options.allow_overwrite {
                OpenOptions::new()
                    .create(true)
                    .truncate(false)
                    .write(true)
                    .open(&full_path)
                    .with_context(|| format!("error creating {full_path:?}"))?;
            } else {
                OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(&full_path)
                    .with_context(|| {
                        format!(
                            "error creating a new file (because allow_overwrite = false) {:?}",
                            &full_path
                        )
                    })?;
            }
            files.push(Arc::new(OpenedFile::new_closed(
                full_path,
                self.pool.clone(),
            )));
        }

        self.opened_files = files;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::filesystem::opened_file::OpenedFile;

    fn storage_with_files(dir: &Path, n: usize, cap: usize) -> FilesystemStorage {
        let pool = FilePool::new(cap);
        let opened_files = (0..n)
            .map(|i| {
                let p = dir.join(format!("f{i}"));
                std::fs::File::create(&p).unwrap();
                Arc::new(OpenedFile::new_closed(p, pool.clone()))
            })
            .collect();
        FilesystemStorage {
            output_folder: dir.to_path_buf(),
            opened_files,
            pool,
        }
    }

    /// 2,000 files through a pool of 16: every write lands, every read
    /// comes back, and the pool never holds more than the cap.
    #[test]
    fn pool_bounds_open_handles() {
        let td = tempfile::TempDir::with_prefix("pool").unwrap();
        let s = storage_with_files(td.path(), 2000, 16);
        for i in 0..2000 {
            let byte = u8::try_from(i % 256).unwrap();
            s.pwrite_all(i, 0, &[byte; 4]).unwrap();
            assert!(
                s.pool.open_count() <= 16,
                "open={} after write {i}",
                s.pool.open_count()
            );
        }
        let mut buf = [0u8; 4];
        for i in (0..2000).rev() {
            s.pread_exact(i, 0, &mut buf).unwrap();
            assert_eq!(buf, [u8::try_from(i % 256).unwrap(); 4]);
            assert!(
                s.pool.open_count() <= 16,
                "open={} after read {i}",
                s.pool.open_count()
            );
        }
        let open = s.opened_files.iter().filter(|f| f.is_open()).count();
        assert_eq!(open, s.pool.open_count());
        assert!(open <= 16);
    }

    /// A handle in use is never evicted: the guard held across other opens
    /// keeps its file readable.
    #[test]
    fn eviction_skips_a_busy_handle() {
        let td = tempfile::TempDir::with_prefix("busy").unwrap();
        let s = storage_with_files(td.path(), 40, 4);
        s.pwrite_all(0, 0, b"keep").unwrap();
        let held = s.opened_files[0].lock_read().unwrap();
        for i in 1..40 {
            s.pwrite_all(i, 0, b"x").unwrap();
        }
        assert!(
            s.opened_files[0].is_open(),
            "busy handle must survive eviction"
        );
        let mut buf = [0u8; 4];
        held.pread_exact(0, &mut buf).unwrap();
        assert_eq!(&buf, b"keep");
    }

    /// The pool is shared: two storages together stay under one cap, and a
    /// storage with nothing open still gets room by closing the other's.
    #[test]
    fn pool_is_shared_across_storages() {
        let td = tempfile::TempDir::with_prefix("shared").unwrap();
        let pool = FilePool::new(8);
        let mk = |name: &str| {
            let dir = td.path().join(name);
            std::fs::create_dir_all(&dir).unwrap();
            let opened_files = (0..50)
                .map(|i| {
                    let p = dir.join(format!("f{i}"));
                    std::fs::File::create(&p).unwrap();
                    Arc::new(OpenedFile::new_closed(p, pool.clone()))
                })
                .collect();
            FilesystemStorage {
                output_folder: dir,
                opened_files,
                pool: pool.clone(),
            }
        };
        let (a, b) = (mk("a"), mk("b"));
        for i in 0..50 {
            a.pwrite_all(i, 0, b"a").unwrap();
        }
        assert_eq!(pool.open_count(), 8);
        for i in 0..50 {
            b.pwrite_all(i, 0, b"b").unwrap();
            assert!(pool.open_count() <= 8, "open={}", pool.open_count());
        }
        let open_in_a = a.opened_files.iter().filter(|f| f.is_open()).count();
        assert_eq!(open_in_a, 0, "b's opens must have evicted a's idle handles");
    }

    /// `take` moves the entries out: the old storage never reopens them.
    #[test]
    fn taken_storage_does_not_reopen() {
        let td = tempfile::TempDir::with_prefix("taken").unwrap();
        let s = storage_with_files(td.path(), 3, 4);
        s.pwrite_all(1, 0, b"abcd").unwrap();
        let taken = s.take().unwrap();
        let mut buf = [0u8; 4];
        assert!(
            s.pread_exact(1, 0, &mut buf).is_err(),
            "old storage must refuse"
        );
        taken.pread_exact(1, 0, &mut buf).unwrap();
        assert_eq!(&buf, b"abcd");
    }
}
