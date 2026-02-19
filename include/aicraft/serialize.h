/**
 * @file serialize.h
 * @brief Model serialization — binary save/load for tensors, layers, and full models.
 *
 * Provides a compact binary format identified by a magic number and version tag,
 * followed by tensor metadata and raw float data.
 *
 * @verbatim
 * File format:
 *   [4 bytes] Magic: "ACML"
 *   [4 bytes] Version: uint32
 *   [4 bytes] Num tensors: uint32
 *   For each tensor:
 *     [4 bytes]              ndim
 *     [ndim * 8 bytes]       dims (as uint64)
 *     [4 bytes]              total_size
 *     [total_size * 4 bytes] float data
 * @endverbatim
 */

#ifndef AICRAFT_SERIALIZE_H
#define AICRAFT_SERIALIZE_H

#include "aicraft/tensor.h"
#include "aicraft/layers.h"
#include "aicraft/error.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup serialize Model Serialization
 * @brief Binary save/load utilities for tensors, layers, and complete models.
 * @{
 */

/** @brief Magic number identifying an AICRAFT model file ("ACML" in ASCII). */
#define AC_SERIALIZE_MAGIC   0x4C4D4341  /* "ACML" */

/** @brief Current serialization format version. */
#define AC_SERIALIZE_VERSION 1

/** @name Single Tensor I/O
 * @{ */

/**
 * @brief Write a single tensor to an open binary file.
 *
 * Serializes the tensor's dimensionality, shape, and float data.
 *
 * @param t   Pointer to the tensor to save. Must not be NULL.
 * @param fp  Open file handle in binary-write mode. Must not be NULL.
 * @return AC_OK on success, or an appropriate ac_error_code on failure.
 * @see ac_tensor_load
 */
static AC_INLINE ac_error_code ac_tensor_save(const ac_tensor* t, FILE* fp) {
    AC_CHECK_RET(t != NULL, AC_ERR_NULL_PTR, "Cannot save NULL tensor");
    AC_CHECK_RET(fp != NULL, AC_ERR_NULL_PTR, "Cannot save to NULL file");
    
    uint32_t ndim = (uint32_t)t->shape.ndim;
    if (fwrite(&ndim, sizeof(uint32_t), 1, fp) != 1) 
        AC_ERROR_RET(AC_ERR_FILE_IO, "Failed to write ndim");
    
    for (uint32_t i = 0; i < ndim; i++) {
        uint64_t dim = (uint64_t)t->shape.dims[i];
        if (fwrite(&dim, sizeof(uint64_t), 1, fp) != 1)
            AC_ERROR_RET(AC_ERR_FILE_IO, "Failed to write dim %u", i);
    }
    
    uint64_t total = (uint64_t)t->shape.total_size;
    if (fwrite(&total, sizeof(uint64_t), 1, fp) != 1)
        AC_ERROR_RET(AC_ERR_FILE_IO, "Failed to write total_size");
    
    if (fwrite(t->data, sizeof(float), (size_t)total, fp) != (size_t)total)
        AC_ERROR_RET(AC_ERR_FILE_IO, "Failed to write tensor data");
    
    return AC_OK;
}

/**
 * @brief Read a single tensor from an open binary file.
 *
 * Allocates a new tensor and fills it with the data read from @p fp.
 *
 * @param fp  Open file handle in binary-read mode. Must not be NULL.
 * @return Pointer to the newly created tensor, or NULL on failure.
 * @note The caller is responsible for freeing the returned tensor.
 * @see ac_tensor_save
 */
static AC_INLINE ac_tensor* ac_tensor_load(FILE* fp) {
    AC_CHECK_NULL(fp != NULL, AC_ERR_NULL_PTR, "Cannot load from NULL file");
    
    uint32_t ndim;
    if (fread(&ndim, sizeof(uint32_t), 1, fp) != 1) {
        AC_ERROR_NULL(AC_ERR_FILE_IO, "Failed to read ndim");
    }
    if (ndim > AC_MAX_DIMS) {
        AC_ERROR_NULL(AC_ERR_INVALID_DIM, "ndim=%u exceeds AC_MAX_DIMS=%d", ndim, AC_MAX_DIMS);
    }
    
    ac_size dims[AC_MAX_DIMS];
    for (uint32_t i = 0; i < ndim; i++) {
        uint64_t dim;
        if (fread(&dim, sizeof(uint64_t), 1, fp) != 1) {
            AC_ERROR_NULL(AC_ERR_FILE_IO, "Failed to read dim %u", i);
        }
        dims[i] = (ac_size)dim;
    }
    
    uint64_t total;
    if (fread(&total, sizeof(uint64_t), 1, fp) != 1) {
        AC_ERROR_NULL(AC_ERR_FILE_IO, "Failed to read total_size");
    }
    
    ac_shape shape = ac_shape_make(ndim, dims);
    if ((uint64_t)shape.total_size != total) {
        AC_ERROR_NULL(AC_ERR_SHAPE_MISMATCH, "Shape total %zu != stored %llu",
                      shape.total_size, (unsigned long long)total);
    }
    
    ac_tensor* t = ac_tensor_create(shape, 0);
    if (!t) {
        AC_ERROR_NULL(AC_ERR_OUT_OF_MEMORY, "Failed to create tensor for loading");
    }
    
    if (fread(t->data, sizeof(float), (size_t)total, fp) != (size_t)total) {
        AC_ERROR_NULL(AC_ERR_FILE_IO, "Failed to read tensor data");
    }
    
    return t;
}

/** @} */

/** @name Model I/O
 * @{ */

/**
 * @brief Save a complete model (parameter group) to a binary file.
 *
 * Writes the ACML header followed by every tensor in the parameter group.
 *
 * @param path    File path to write to. Must not be NULL.
 * @param params  Pointer to the parameter group to save. Must not be NULL.
 * @return AC_OK on success, or an appropriate ac_error_code on failure.
 * @see ac_model_load
 */
static AC_INLINE ac_error_code ac_model_save(const char* path, 
                                              const ac_param_group* params) {
    AC_CHECK_RET(path != NULL, AC_ERR_NULL_PTR, "Cannot save to NULL path");
    AC_CHECK_RET(params != NULL, AC_ERR_NULL_PTR, "Cannot save NULL params");
    
    FILE* fp = fopen(path, "wb");
    if (!fp) AC_ERROR_RET(AC_ERR_FILE_IO, "Cannot open '%s' for writing", path);
    
    /* Write header */
    uint32_t magic = AC_SERIALIZE_MAGIC;
    uint32_t version = AC_SERIALIZE_VERSION;
    uint32_t num_tensors = (uint32_t)params->num_params;
    
    if (fwrite(&magic, sizeof(uint32_t), 1, fp) != 1 ||
        fwrite(&version, sizeof(uint32_t), 1, fp) != 1 ||
        fwrite(&num_tensors, sizeof(uint32_t), 1, fp) != 1) {
        fclose(fp);
        AC_ERROR_RET(AC_ERR_FILE_IO, "Failed to write header");
    }
    
    /* Write each parameter tensor */
    for (uint32_t i = 0; i < num_tensors; i++) {
        ac_error_code err = ac_tensor_save(params->params[i], fp);
        if (err != AC_OK) {
            fclose(fp);
            return err;
        }
    }
    
    fclose(fp);
    return AC_OK;
}

/**
 * @brief Load model parameters from a binary file into an existing parameter group.
 *
 * Validates the header (magic, version, tensor count) and reads float data
 * directly into the tensors already present in @p params.
 *
 * @param path    File path to read from. Must not be NULL.
 * @param params  Pointer to the parameter group to populate. Must not be NULL.
 *                Its tensor shapes must match those stored in the file.
 * @return AC_OK on success, or an appropriate ac_error_code on failure.
 * @note The parameter group must be pre-allocated with the correct tensor shapes.
 * @see ac_model_save
 */
static AC_INLINE ac_error_code ac_model_load(const char* path, 
                                              ac_param_group* params) {
    AC_CHECK_RET(path != NULL, AC_ERR_NULL_PTR, "Cannot load from NULL path");
    AC_CHECK_RET(params != NULL, AC_ERR_NULL_PTR, "Cannot load into NULL params");
    
    FILE* fp = fopen(path, "rb");
    if (!fp) AC_ERROR_RET(AC_ERR_FILE_IO, "Cannot open '%s' for reading", path);
    
    /* Read and validate header */
    uint32_t magic, version, num_tensors;
    if (fread(&magic, sizeof(uint32_t), 1, fp) != 1 ||
        fread(&version, sizeof(uint32_t), 1, fp) != 1 ||
        fread(&num_tensors, sizeof(uint32_t), 1, fp) != 1) {
        fclose(fp);
        AC_ERROR_RET(AC_ERR_FILE_IO, "Failed to read header");
    }
    
    if (magic != AC_SERIALIZE_MAGIC) {
        fclose(fp);
        AC_ERROR_RET(AC_ERR_INVALID_ARGUMENT, "Invalid file magic: 0x%08X", magic);
    }
    
    if (version > AC_SERIALIZE_VERSION) {
        fclose(fp);
        AC_ERROR_RET(AC_ERR_INVALID_ARGUMENT, "Unsupported version: %u", version);
    }
    
    if (num_tensors != (uint32_t)params->num_params) {
        fclose(fp);
        AC_ERROR_RET(AC_ERR_SHAPE_MISMATCH, 
                     "Model has %u params, file has %u", 
                     (uint32_t)params->num_params, num_tensors);
    }
    
    /* Load each parameter tensor's data into existing tensors */
    for (uint32_t i = 0; i < num_tensors; i++) {
        uint32_t ndim;
        if (fread(&ndim, sizeof(uint32_t), 1, fp) != 1) {
            fclose(fp);
            AC_ERROR_RET(AC_ERR_FILE_IO, "Failed to read param %u ndim", i);
        }
        
        /* Read and validate dims */
        ac_size dims[AC_MAX_DIMS];
        for (uint32_t d = 0; d < ndim; d++) {
            uint64_t dim;
            if (fread(&dim, sizeof(uint64_t), 1, fp) != 1) {
                fclose(fp);
                AC_ERROR_RET(AC_ERR_FILE_IO, "Failed to read param %u dim %u", i, d);
            }
            dims[d] = (ac_size)dim;
        }
        
        uint64_t total;
        if (fread(&total, sizeof(uint64_t), 1, fp) != 1) {
            fclose(fp);
            AC_ERROR_RET(AC_ERR_FILE_IO, "Failed to read param %u total", i);
        }
        
        /* Validate shape matches */
        if (total != (uint64_t)params->params[i]->shape.total_size) {
            fclose(fp);
            AC_ERROR_RET(AC_ERR_SHAPE_MISMATCH,
                         "Param %u: expected size %zu, got %u",
                         i, params->params[i]->shape.total_size, total);
        }
        
        /* Read data directly into existing tensor */
        if (fread(params->params[i]->data, sizeof(float), total, fp) != total) {
            fclose(fp);
            AC_ERROR_RET(AC_ERR_FILE_IO, "Failed to read param %u data", i);
        }
    }
    
    fclose(fp);
    return AC_OK;
}

/** @} */

/** @name Dense Layer I/O
 * @{ */

/**
 * @brief Save a dense layer's weight and bias tensors to an open file.
 *
 * @param layer  Pointer to the dense layer. Must not be NULL.
 * @param fp     Open file handle in binary-write mode.
 * @return AC_OK on success, or an appropriate ac_error_code on failure.
 * @see ac_dense_load
 */
static AC_INLINE ac_error_code ac_dense_save(const ac_dense* layer, FILE* fp) {
    ac_error_code err = ac_tensor_save(layer->weight, fp);
    if (err != AC_OK) return err;
    return ac_tensor_save(layer->bias, fp);
}

/**
 * @brief Load weight and bias data into an existing dense layer from an open file.
 *
 * @param layer  Pointer to the dense layer whose tensors will be filled.
 * @param fp     Open file handle in binary-read mode.
 * @return AC_OK on success, or an appropriate ac_error_code on failure.
 * @note The layer must already be initialised with correctly shaped tensors.
 * @see ac_dense_save
 */
static AC_INLINE ac_error_code ac_dense_load(ac_dense* layer, FILE* fp) {
    /* Load weight data */
    uint32_t ndim;
    if (fread(&ndim, sizeof(uint32_t), 1, fp) != 1)
        AC_ERROR_RET(AC_ERR_FILE_IO, "Failed to read weight ndim");
    
    for (uint32_t i = 0; i < ndim; i++) {
        uint64_t dim;
        if (fread(&dim, sizeof(uint64_t), 1, fp) != 1)
            AC_ERROR_RET(AC_ERR_FILE_IO, "Failed to read weight dim");
    }
    
    uint64_t total;
    if (fread(&total, sizeof(uint64_t), 1, fp) != 1)
        AC_ERROR_RET(AC_ERR_FILE_IO, "Failed to read weight total");
    
    if (total != (uint64_t)layer->weight->shape.total_size)
        AC_ERROR_RET(AC_ERR_SHAPE_MISMATCH, "Weight size mismatch");
    
    if (fread(layer->weight->data, sizeof(float), total, fp) != total)
        AC_ERROR_RET(AC_ERR_FILE_IO, "Failed to read weight data");
    
    /* Load bias data */
    if (fread(&ndim, sizeof(uint32_t), 1, fp) != 1)
        AC_ERROR_RET(AC_ERR_FILE_IO, "Failed to read bias ndim");
    
    for (uint32_t i = 0; i < ndim; i++) {
        uint64_t dim;
        if (fread(&dim, sizeof(uint64_t), 1, fp) != 1)
            AC_ERROR_RET(AC_ERR_FILE_IO, "Failed to read bias dim");
    }
    
    if (fread(&total, sizeof(uint64_t), 1, fp) != 1)
        AC_ERROR_RET(AC_ERR_FILE_IO, "Failed to read bias total");
    
    if (total != (uint64_t)layer->bias->shape.total_size)
        AC_ERROR_RET(AC_ERR_SHAPE_MISMATCH, "Bias size mismatch");
    
    if (fread(layer->bias->data, sizeof(float), total, fp) != total)
        AC_ERROR_RET(AC_ERR_FILE_IO, "Failed to read bias data");
    
    return AC_OK;
}

/** @} */
/** @} */

#ifdef __cplusplus
}
#endif

#endif /* AICRAFT_SERIALIZE_H */
