import os

# PyTorch and faiss-cpu wheels each bundle their own libomp on macOS; loading both in one process (query
# embedding + ANN search) aborts with "OMP: Error #15" unless duplicates are allowed. Results are checked
# against exact search in benchmarks/bench_search.py.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
