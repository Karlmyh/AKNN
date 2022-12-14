cimport numpy as cnp
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.intp_t ITYPE_t

cdef int partition_node_indices(
        DTYPE_t *data,
        ITYPE_t *node_indices,
        ITYPE_t split_dim,
        ITYPE_t split_index,
        ITYPE_t n_features,
        ITYPE_t n_points) except -1
