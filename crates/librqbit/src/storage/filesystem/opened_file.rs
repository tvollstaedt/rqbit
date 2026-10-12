use std::{
    collections::VecDeque,
    fs::{File, OpenOptions},
    io::IoSlice,
    ops::{Deref, DerefMut},
    path::PathBuf,
    sync::{
        Arc, OnceLock, Weak,
        atomic::{AtomicUsize, Ordering},
    },
};

use anyhow::Context;
use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::Error;

pub trait OurFileExt {
    fn pwrite_all_vectored(&self, offset: u64, bufs: [IoSlice<'_>; 2]) -> anyhow::Result<usize>;
    fn pread_exact(&self, offset: u64, buf: &mut [u8]) -> anyhow::Result<()>;
    fn pwrite_all(&self, offset: u64, buf: &[u8]) -> anyhow::Result<()>;
}

impl OurFileExt for File {
    #[cfg(unix)]
    fn pwrite_all_vectored(&self, offset: u64, bufs: [IoSlice<'_>; 2]) -> anyhow::Result<usize> {
        nix::sys::uio::pwritev(self, &bufs, offset.try_into()?).context("error calling pwritev")
    }

    #[cfg(not(unix))]
    fn pwrite_all_vectored(&self, offset: u64, bufs: [IoSlice<'_>; 2]) -> anyhow::Result<usize> {
        match (bufs[0].len(), bufs[1].len()) {
            (len, 0) if len > 0 => {
                self.pwrite_all(offset, &bufs[0])?;
                Ok(len)
            }
            (0, len) if len > 0 => {
                self.pwrite_all(offset, &bufs[1])?;
                Ok(len)
            }
            (0, 0) => Ok(0),
            (l0, l1) => {
                // concatenate the buffers in memory so that we issue one write call instead of 2
                // assumes the message is <= CHUNK_SIZE
                use librqbit_core::constants::CHUNK_SIZE;
                let mut buf = [0u8; CHUNK_SIZE as usize];

                buf.get_mut(..l0)
                    .context("buf too small")?
                    .copy_from_slice(&bufs[0]);
                buf.get_mut(l0..l0 + l1)
                    .context("buf too small")?
                    .copy_from_slice(&bufs[1]);
                self.pwrite_all(offset, &buf[..l0 + l1])?;
                Ok(l0 + l1)
            }
        }
    }

    #[cfg(unix)]
    fn pread_exact(&self, offset: u64, buf: &mut [u8]) -> anyhow::Result<()> {
        use std::os::unix::fs::FileExt;

        Ok(self.read_exact_at(buf, offset)?)
    }

    #[cfg(windows)]
    fn pread_exact(&self, offset: u64, buf: &mut [u8]) -> anyhow::Result<()> {
        use std::os::windows::fs::FileExt;
        self.seek_read(buf, offset)?;
        Ok(())
    }

    #[cfg(not(any(windows, unix)))]
    fn pread_exact(&self, offset: u64, buf: &mut [u8]) -> anyhow::Result<()> {
        anyhow::bail!("pread_exact not implemented for your platform")
    }

    #[cfg(unix)]
    fn pwrite_all(&self, offset: u64, buf: &[u8]) -> anyhow::Result<()> {
        use std::os::unix::fs::FileExt;
        Ok(self.write_all_at(buf, offset)?)
    }

    #[cfg(windows)]
    fn pwrite_all(&self, offset: u64, buf: &[u8]) -> anyhow::Result<()> {
        use std::os::windows::fs::FileExt;

        let mut remaining = buf.len();
        let mut buf = buf;
        let mut offset = offset;
        while remaining > 0 {
            let written = self.seek_write(&buf[..remaining], offset)?;
            remaining -= written;
            offset += written as u64;
            buf = &buf[written..];
        }
        Ok(())
    }

    #[cfg(not(any(windows, unix)))]
    fn pwrite_all(&self, offset: u64, buf: &[u8]) -> anyhow::Result<()> {
        anyhow::bail!("pwrite_all not implemented for your platform")
    }
}

/// Bounds the handles open across every storage in the process: `cap` at
/// most, the oldest closed first. A torrent has one entry per file, and a
/// 14,000-file torrent would otherwise hold 14,000 fds from `init` on.
#[derive(Debug)]
pub(crate) struct FilePool {
    cap: usize,
    open: AtomicUsize,
    order: Mutex<VecDeque<Weak<OpenedFile>>>,
}

impl FilePool {
    pub fn new(cap: usize) -> Arc<Self> {
        Arc::new(Self {
            cap: cap.max(1),
            open: AtomicUsize::new(0),
            order: Mutex::new(VecDeque::new()),
        })
    }

    /// One pool for the process, sized from the fd limit at first use: a
    /// quarter of the soft limit, between 64 and 2,048. The rest is for
    /// sockets, which burst to a few hundred when the torrents go live.
    pub fn global() -> Arc<Self> {
        static GLOBAL: OnceLock<Arc<FilePool>> = OnceLock::new();
        GLOBAL
            .get_or_init(|| {
                #[cfg(unix)]
                let soft = rlimit::getrlimit(rlimit::Resource::NOFILE)
                    .map(|(soft, _)| soft)
                    .unwrap_or(1024);
                // No descriptor limit on Windows; the pool still bounds the handles.
                #[cfg(not(unix))]
                let soft: u64 = 8192;
                let cap = usize::try_from(soft / 4).unwrap_or(2048).clamp(64, 2048);
                tracing::debug!(cap, soft, "file pool sized");
                Self::new(cap)
            })
            .clone()
    }

    pub fn open_count(&self) -> usize {
        self.open.load(Ordering::Relaxed)
    }

    /// Close the oldest idle handles until one more fits. Busy handles go
    /// back to the end of the queue; taken or dropped entries fall out.
    fn make_room(&self) {
        if self.open_count() < self.cap {
            return;
        }
        let candidates: Vec<Weak<OpenedFile>> = {
            let mut order = self.order.lock();
            let n = (self.open_count() + 1 - self.cap + 8).min(order.len());
            order.drain(..n).collect()
        };
        for weak in candidates {
            let Some(of) = weak.upgrade() else { continue };
            if (self.open_count() < self.cap || !of.try_close()) && of.is_open() {
                self.order.lock().push_back(weak);
            }
        }
    }

    fn opened(&self, of: &Arc<OpenedFile>) {
        self.open.fetch_add(1, Ordering::Relaxed);
        self.order.lock().push_back(Arc::downgrade(of));
    }

    fn closed(&self) {
        self.open.fetch_sub(1, Ordering::Relaxed);
    }
}

#[derive(Debug)]
enum Fd {
    /// Known path, no handle: opened on first access.
    Closed(PathBuf),
    Open(PathBuf, File),
    /// Moved out by `take_clone`, or a padding file: never opened again.
    Taken,
}

impl Fd {
    fn file(&self) -> Option<&File> {
        match self {
            Fd::Open(_, f) => Some(f),
            _ => None,
        }
    }

    fn file_mut(&mut self) -> Option<&mut File> {
        match self {
            Fd::Open(_, f) => Some(f),
            _ => None,
        }
    }

    #[cfg(windows)]
    fn path(&self) -> Option<&PathBuf> {
        match self {
            Fd::Closed(p) | Fd::Open(p, _) => Some(p),
            Fd::Taken => None,
        }
    }
}

#[derive(Debug)]
struct OpenedFileLocked {
    fd: Fd,
    #[cfg(windows)]
    tried_marking_sparse: bool,
}

#[derive(Debug)]
pub(crate) struct OpenedFile {
    file: RwLock<OpenedFileLocked>,
    pool: Arc<FilePool>,
}

impl OpenedFile {
    /// An entry whose file exists on disk; the handle is opened on first use.
    pub fn new_closed(path: PathBuf, pool: Arc<FilePool>) -> Self {
        Self {
            file: RwLock::new(OpenedFileLocked {
                fd: Fd::Closed(path),
                #[cfg(windows)]
                tried_marking_sparse: false,
            }),
            pool,
        }
    }

    pub fn new_dummy(pool: Arc<FilePool>) -> Self {
        Self {
            file: RwLock::new(OpenedFileLocked {
                fd: Fd::Taken,
                #[cfg(windows)]
                tried_marking_sparse: false,
            }),
            pool,
        }
    }

    pub fn take_clone(&self) -> anyhow::Result<Self> {
        let mut g = self.file.write();
        let fd = std::mem::replace(&mut g.fd, Fd::Taken);
        Ok(Self {
            file: RwLock::new(OpenedFileLocked {
                fd,
                #[cfg(windows)]
                tried_marking_sparse: g.tried_marking_sparse,
            }),
            pool: self.pool.clone(),
        })
    }

    /// Close the handle unless someone is using it. `true` when closed.
    pub fn try_close(&self) -> bool {
        let Some(mut g) = self.file.try_write() else {
            return false;
        };
        if let Fd::Open(path, _) = &g.fd {
            let path = path.clone();
            g.fd = Fd::Closed(path);
            self.pool.closed();
            return true;
        }
        false
    }

    pub fn is_open(&self) -> bool {
        matches!(self.file.read().fd, Fd::Open(..))
    }

    /// Open the handle if it is closed. Under the write lock, so a reader
    /// never sees a half-open entry and eviction cannot interleave.
    fn open_locked(self: &Arc<Self>, g: &mut OpenedFileLocked) -> crate::Result<()> {
        match &g.fd {
            Fd::Open(..) => Ok(()),
            Fd::Taken => Err(Error::FsFileIsNone),
            Fd::Closed(path) => {
                self.pool.make_room();
                let f = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(path)
                    .with_context(|| format!("error opening {path:?} in read/write mode"))
                    .map_err(Error::Anyhow)?;
                let path = path.clone();
                g.fd = Fd::Open(path, f);
                self.pool.opened(self);
                Ok(())
            }
        }
    }

    /// The entry with its handle open, under the read lock.
    fn ensure_open(self: &Arc<Self>) -> crate::Result<RwLockReadGuard<'_, OpenedFileLocked>> {
        {
            let g = self.file.read();
            match g.fd {
                Fd::Open(..) => return Ok(g),
                Fd::Taken => return Err(Error::FsFileIsNone),
                Fd::Closed(_) => {}
            }
        }
        let mut g = self.file.write();
        self.open_locked(&mut g)?;
        Ok(RwLockWriteGuard::downgrade(g))
    }

    pub fn lock_read(self: &Arc<Self>) -> crate::Result<impl Deref<Target = File> + '_> {
        let g = self.ensure_open()?;
        Ok(RwLockReadGuard::map(g, |l| {
            l.fd.file().expect("ensure_open leaves the fd open")
        }))
    }

    #[allow(dead_code)]
    pub fn lock_write(self: &Arc<Self>) -> crate::Result<impl DerefMut<Target = File> + '_> {
        let mut g = self.file.write();
        self.open_locked(&mut g)?;
        Ok(RwLockWriteGuard::map(g, |l| {
            l.fd.file_mut().expect("open_locked leaves the fd open")
        }))
    }

    #[cfg(windows)]
    pub fn try_mark_sparse(self: &Arc<Self>) -> crate::Result<impl Deref<Target = File> + '_> {
        {
            let g = self.ensure_open()?;
            if g.tried_marking_sparse {
                return Ok(RwLockReadGuard::map(g, |l| {
                    l.fd.file().expect("ensure_open leaves the fd open")
                }));
            }
        }
        let mut g = self.file.write();
        self.open_locked(&mut g)?;
        if !g.tried_marking_sparse {
            g.tried_marking_sparse = true;
            let f = g.fd.file().expect("open_locked leaves the fd open");
            // Not inside the log macro: tracing evaluates fields only when
            // the event is enabled, and the default level is info.
            if super::sparse::mark_file_sparse(f) {
                tracing::debug!(path=?g.fd.path(), "marked sparse");
            } else {
                let err = std::io::Error::last_os_error();
                tracing::warn!(path=?g.fd.path(), ?err, "could not mark file sparse; writes past EOF will allocate the gap");
            }
        }
        let g = parking_lot::RwLockWriteGuard::downgrade(g);
        Ok(RwLockReadGuard::map(g, |l| {
            l.fd.file().expect("open_locked leaves the fd open")
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use librqbit_core::constants::CHUNK_SIZE;
    use peer_binary_protocol::DoubleBufHelper;
    use tempfile::TempDir;

    use crate::storage::filesystem::opened_file::OurFileExt;

    /// The shipped binary logs at info: marking must not depend on a debug
    /// event being enabled.
    #[cfg(windows)]
    #[test]
    fn marks_file_sparse_without_a_tracing_subscriber() {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_SPARSE_FILE: u32 = 0x200;
        let td = TempDir::with_prefix("sparse").unwrap();
        let path = td.path().join("f");
        std::fs::File::create(&path).unwrap();
        let of = std::sync::Arc::new(super::OpenedFile::new_closed(
            path.clone(),
            super::FilePool::new(4),
        ));
        of.try_mark_sparse()
            .unwrap()
            .pwrite_all(1 << 30, &[1u8; 8])
            .unwrap();
        let attrs = std::fs::metadata(&path).unwrap().file_attributes();
        assert!(
            attrs & FILE_ATTRIBUTE_SPARSE_FILE != 0,
            "file written at 1 GiB is not sparse: attrs={attrs:#x}"
        );
    }

    #[test]
    fn test_pwrite_all_vectored() {
        let td = TempDir::with_prefix("test_pwrite_all_vectored").unwrap();
        let mut tmp_buf = [0u8; CHUNK_SIZE as usize];
        for bufsize in [10000usize, CHUNK_SIZE as usize] {
            let mut buf = vec![0u8; bufsize];
            rand::fill(&mut buf[..]);
            for split_point in [0, bufsize / 2, bufsize] {
                let path = td.path().join(format!("file_{bufsize}_{split_point}"));
                let file = std::fs::OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(&path)
                    .unwrap();
                let (first, second) = buf.split_at(split_point);
                let bufs = DoubleBufHelper::new(first, second).as_ioslices(bufsize);
                file.pwrite_all_vectored(0, bufs).unwrap();

                let mut file = std::fs::File::open(&path).unwrap();
                assert_eq!(file.metadata().unwrap().len(), bufsize as u64, "{path:?}");
                file.read_exact(&mut tmp_buf[..bufsize]).unwrap();
                assert_eq!(&tmp_buf[..bufsize], buf);
            }
        }
    }
}
