#pragma once
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

// --------------------------------------------------------------------
// MappedJournal — Memory-Mapped Sequential Event Log
// --------------------------------------------------------------------
//
// Purpose
// ───────
//   Persist every engine event (new order, trade, cancel, modify) to disk
//   without introducing synchronous I/O latency into the hot path.
//
// Mechanism
// ─────────
//   The journal file is pre-sized to MaxEntries * sizeof(T) bytes and mapped
//   into the process address space with MAP_SHARED.  Appending an entry is a
//   plain memory assignment — the CPU writes to the kernel's page cache and
//   the kernel flushes dirty pages to disk in the background (write-back I/O).
//
//   This avoids:
//     • read() / write() syscall overhead        (~1–3 µs per call)
//     • Kernel-side data copies
//     • Any blocking on disk I/O in the journal writer
//
// Threading model
// ───────────────
//   MappedJournal is NOT thread-safe.  It is intended to be owned and
//   written exclusively by the Publisher thread (Core 3) so that the
//   Engine thread (Core 2) is never burdened with I/O bookkeeping.
//
// Recovery
// ────────
//   On restart the journal can be replayed (via loadSnapshot + journal
//   replay) to reconstruct the book state at the point of failure.
//   For point-in-time recovery, pair with saveSnapshot() on the OrderBook.
//
// Template parameters
// ───────────────────
//   T          — event record type (e.g. UpdateMessage)
//   MaxEntries — pre-allocated capacity; the file is sized at construction

template<typename T, size_t MaxEntries>
class MappedJournal
{
public:
    explicit MappedJournal(const std::string& filename)
    {
        fileSize_ = MaxEntries * sizeof(T);

        fd_ = open(filename.c_str(), O_RDWR | O_CREAT, 0666);
        if (fd_ == -1)
            throw std::runtime_error("MappedJournal: failed to open " + filename);

        if (ftruncate(fd_, static_cast<off_t>(fileSize_)) == -1)
        {
            close(fd_);
            throw std::runtime_error("MappedJournal: failed to size " + filename);
        }

        void* addr = mmap(nullptr, fileSize_,
                          PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (addr == MAP_FAILED)
        {
            close(fd_);
            throw std::runtime_error("MappedJournal: mmap failed for " + filename);
        }

        data_ = static_cast<T*>(addr);

        // Hint to the kernel that pages will be accessed sequentially.
        // The kernel can pre-fetch pages ahead of the write cursor.
        madvise(data_, fileSize_, MADV_SEQUENTIAL);
    }

    ~MappedJournal()
    {
        if (data_ != nullptr)
            munmap(data_, fileSize_);
        if (fd_ != -1)
            close(fd_);
    }

    // Non-copyable and non-movable (owns the fd and mapping).
    MappedJournal(const MappedJournal&)            = delete;
    MappedJournal& operator=(const MappedJournal&) = delete;
    MappedJournal(MappedJournal&&)                 = delete;
    MappedJournal& operator=(MappedJournal&&)      = delete;

    // Append one entry to the journal.  This is a single cache-line write
    // — no syscall, no lock, no copy beyond what sizeof(T) requires.
    // Silently drops entries once the pre-allocated capacity is exhausted.
    inline void append(const T& entry)
    {
        if (writeIdx_ < MaxEntries)
            data_[writeIdx_++] = entry;
    }

    size_t count() const { return writeIdx_; }

private:
    int    fd_       = -1;
    T*     data_     = nullptr;
    size_t fileSize_ = 0;
    size_t writeIdx_ = 0;
};
