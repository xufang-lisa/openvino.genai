// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <list>

#include "openvino/runtime/tensor.hpp"
#include "paged_attention_transformations.hpp"

namespace ov::genai {

class CacheManager {
    size_t m_num_decoder_layers = 0;
    std::string m_device;
    size_t m_block_size = 0; // block size is per inference device 
    std::vector<ov::element::Type> m_key_precisions, m_value_precisions;
    std::vector<ov::PartialShape> m_key_shapes, m_value_shapes;
    std::vector<ov::Tensor> m_key_cache, m_value_cache;
    size_t m_num_allocated_kv_blocks = 0, m_block_size_in_bytes = 0;
    ov::InferRequest m_request;
    size_t m_k_head_size = 0;

    static ov::Shape set_kv_blocks(ov::PartialShape pshape, size_t num_kv_blocks) {
        pshape[0] = num_kv_blocks;
        return pshape.get_shape();
    }

    void update_request_tensor(size_t decoder_layer_id) {
        m_request.set_tensor(std::string("key_cache.") + std::to_string(decoder_layer_id), m_key_cache[decoder_layer_id]);
        m_request.set_tensor(std::string("value_cache.") + std::to_string(decoder_layer_id), m_value_cache[decoder_layer_id]);
    }

public:
    CacheManager(ov::InferRequest request, const std::vector<KVHeadConfig>& kv_cache_config) :
        m_request(request) {
        // extract information about inference device
        ov::CompiledModel compiled_model = request.get_compiled_model();
        std::vector<std::string> execution_devices = compiled_model.get_property(ov::execution_devices);
        OPENVINO_ASSERT(execution_devices.size() == 1, "Contituous batching: execution device is expected to be CPU or GPU, but got ", execution_devices.size(), " devices");
        m_device = execution_devices[0];
        
        // set block_size depending on device
        const size_t cpu_block_size = 32, gpu_block_size = 16;
        const bool is_gpu = m_device.find("GPU") != std::string::npos;
        m_block_size = is_gpu ? gpu_block_size : cpu_block_size;

        // extract information about KV cache precisions and shapes
        size_t kv_input_index = 0;
        for (const auto& input : compiled_model.inputs()) {
            for (auto & name : input.get_names()) {
                auto cache_precision = input.get_element_type();
                ov::PartialShape pshape;

                if (name.find("key_cache.") == 0) {
                    pshape = input.get_partial_shape();
                    m_block_size_in_bytes += pshape[1].get_length() * pshape[2].get_length() * pshape[3].get_length() * cache_precision.size();
                    m_key_shapes.push_back(pshape);
                    m_key_precisions.push_back(cache_precision);
                    break;
                } else if (name.find("value_cache.") == 0) {
                    pshape = input.get_partial_shape();
                    m_block_size_in_bytes += pshape[1].get_length() * pshape[2].get_length() * pshape[3].get_length() * cache_precision.size();
                    m_value_shapes.push_back(pshape);
                    m_value_precisions.push_back(cache_precision);
                    ++kv_input_index;
                    break;
                }
            }
        }

        m_num_decoder_layers = m_value_precisions.size();
        OPENVINO_ASSERT(m_num_decoder_layers == m_key_precisions.size(), "Invalid case: a different number of K and V caches in a LLM model");
    }

    size_t get_num_decoder_layers() const {
        return m_num_decoder_layers;
    }

    std::string get_device() const {
        return m_device;
    }

    size_t get_block_size() const {
        return m_block_size;
    }

    ov::element::Type get_key_cache_precision(size_t decoder_layer_id) const {
        OPENVINO_ASSERT(decoder_layer_id < m_key_precisions.size());
        return m_key_precisions[decoder_layer_id];
    }

    ov::element::Type get_value_cache_precision(size_t decoder_layer_id) const {
        OPENVINO_ASSERT(decoder_layer_id < m_value_precisions.size());
        return m_value_precisions[decoder_layer_id];
    }

    size_t get_block_size_in_bytes() const {
        return m_block_size_in_bytes;
    }

    void allocate_cache_if_needed(size_t num_kv_blocks) {
        if (m_num_allocated_kv_blocks >= num_kv_blocks) {
            return;
        }

        m_num_allocated_kv_blocks = num_kv_blocks;

        ov::Coordinate start_key{0,0,0,0};
        ov::Coordinate start_value{0,0,0,0};

        if (m_device.find("GPU") == std::string::npos) {// Allocate KV caches
            for (size_t decoder_layer_id = 0; decoder_layer_id < m_num_decoder_layers; ++decoder_layer_id) {
                ov::Shape value_cache_shape = set_kv_blocks(m_value_shapes[decoder_layer_id], num_kv_blocks);
                ov::Shape key_cache_shape = set_kv_blocks(m_key_shapes[decoder_layer_id], num_kv_blocks);

                ov::element::Type key_precision = get_key_cache_precision(decoder_layer_id);
                ov::element::Type value_precision = get_value_cache_precision(decoder_layer_id);

                ov::Tensor key_cache(key_precision, key_cache_shape);
                ov::Tensor value_cache(value_precision, value_cache_shape);

                auto key_cache_roi_end = static_cast<unsigned char*>(key_cache.data());
                auto value_cache_roi_end = static_cast<unsigned char*>(value_cache.data());
                size_t key_roi_size_byte = 0;
                size_t value_roi_size_byte = 0;

                if (m_key_cache.size() > decoder_layer_id) {
                    ov::Coordinate end_key = m_key_cache[decoder_layer_id].get_shape();
                    ov::Coordinate end_value = m_value_cache[decoder_layer_id].get_shape();

                    key_roi_size_byte = m_key_cache[decoder_layer_id].get_byte_size();
                    value_roi_size_byte = m_value_cache[decoder_layer_id].get_byte_size();
                    key_cache_roi_end = static_cast<unsigned char*>(key_cache.data()) + key_roi_size_byte;
                    value_cache_roi_end = static_cast<unsigned char*>(value_cache.data()) + value_roi_size_byte;
                    
                    // copy current cache data
                    ov::Tensor dst_key_roi(key_cache, start_key, end_key);
                    ov::Tensor dst_value_roi(value_cache, start_value, end_value);

                    m_key_cache[decoder_layer_id].copy_to(dst_key_roi);
                    m_value_cache[decoder_layer_id].copy_to(dst_value_roi);

                }

                // set new cache tensors
                if (m_key_cache.size() > decoder_layer_id) {
                    m_key_cache[decoder_layer_id] = key_cache;
                    m_value_cache[decoder_layer_id] = value_cache;
                } else {
                    m_key_cache.emplace_back(key_cache);
                    m_value_cache.emplace_back(value_cache);
                }

                update_request_tensor(decoder_layer_id);
            }
        } else {
            auto remote_context = m_request.get_compiled_model().get_context();

            for (size_t decoder_layer_id = 0; decoder_layer_id < m_num_decoder_layers; ++decoder_layer_id) {
                ov::Shape value_cache_shape = set_kv_blocks(m_value_shapes[decoder_layer_id], num_kv_blocks);
                ov::Shape key_cache_shape = set_kv_blocks(m_key_shapes[decoder_layer_id], num_kv_blocks);

                ov::Tensor key_cache = remote_context.create_tensor(get_key_cache_precision(decoder_layer_id), key_cache_shape);
                ov::Tensor value_cache = remote_context.create_tensor(get_value_cache_precision(decoder_layer_id), value_cache_shape);

                if (m_key_cache.size() > decoder_layer_id) {
                    ov::Coordinate end_key = m_key_cache[decoder_layer_id].get_shape();
                    ov::Coordinate end_value = m_value_cache[decoder_layer_id].get_shape();

                    // copy current cache data
                    ov::RemoteTensor dst_key_roi(key_cache, start_key, end_key);
                    ov::RemoteTensor dst_value_roi(value_cache, start_value, end_value);
                    dst_key_roi.copy_from(m_key_cache[decoder_layer_id]);
                    dst_value_roi.copy_from(m_value_cache[decoder_layer_id]);

                    m_key_cache[decoder_layer_id] = key_cache;
                    m_value_cache[decoder_layer_id] = value_cache;
                } else {
                    m_key_cache.emplace_back(key_cache);
                    m_value_cache.emplace_back(value_cache);
                }

                update_request_tensor(decoder_layer_id);
            }
        }
    }

    ov::Tensor get_key_cache(size_t decoder_layer_id) const {
        OPENVINO_ASSERT(decoder_layer_id < m_key_cache.size(), "decoder_layer_id = ", decoder_layer_id, ", num_layers = ", m_key_cache.size());
        return m_key_cache[decoder_layer_id];
    }

    ov::Tensor get_value_cache(size_t decoder_layer_id) const {
        OPENVINO_ASSERT(decoder_layer_id < m_value_cache.size(), "decoder_layer_id = ", decoder_layer_id, ", num_layers = ", m_value_cache.size());
        return m_value_cache[decoder_layer_id];
    }

    size_t get_v_head_size(size_t layer_id) const {
        return m_value_shapes[layer_id][3].get_length();
    }

    void copy_blocks(const std::map<size_t, std::list<size_t>>& block_copy_map) {
        for (const auto & blocks_pair : block_copy_map) {
            size_t src_block_id = blocks_pair.first;
            const std::list<size_t>& dst_block_ids = blocks_pair.second;
            for (size_t dst_block_id : dst_block_ids) {
                for (size_t decoder_layer_id = 0; decoder_layer_id < m_num_decoder_layers; ++decoder_layer_id) {
                    ov::Shape key_shape = set_kv_blocks(m_key_shapes[decoder_layer_id], m_num_allocated_kv_blocks);
                    ov::Shape value_shape = set_kv_blocks(m_value_shapes[decoder_layer_id], m_num_allocated_kv_blocks);
                    ov::Coordinate key_src_start_roi(key_shape.size(), 0);
                    ov::Coordinate key_src_end_roi = key_shape;
                    ov::Coordinate key_dst_start_roi(key_shape.size(), 0);
                    ov::Coordinate key_dst_end_roi = key_shape;
            
                    ov::Coordinate value_src_start_roi(value_shape.size(), 0);
                    ov::Coordinate value_src_end_roi = value_shape;
                    ov::Coordinate value_dst_start_roi(value_shape.size(), 0);
                    ov::Coordinate value_dst_end_roi = value_shape;
                    key_src_end_roi[0] = (key_src_start_roi[0] = src_block_id) + 1;
                    value_src_end_roi[0] = (value_src_start_roi[0] = src_block_id) + 1;
                    key_dst_end_roi[0] = (key_dst_start_roi[0] = dst_block_id) + 1;
                    value_dst_end_roi[0] = (value_dst_start_roi[0] = dst_block_id) + 1;

                    ov::Tensor key_src_cache_roi(m_key_cache[decoder_layer_id], key_src_start_roi, key_src_end_roi);
                    ov::Tensor key_dst_cache_roi(m_key_cache[decoder_layer_id], key_dst_start_roi, key_dst_end_roi);

                    ov::Tensor value_src_cache_roi(m_value_cache[decoder_layer_id], value_src_start_roi, value_src_end_roi);
                    ov::Tensor value_dst_cache_roi(m_value_cache[decoder_layer_id], value_dst_start_roi, value_dst_end_roi);

                    key_src_cache_roi.copy_to(key_dst_cache_roi);
                    value_src_cache_roi.copy_to(value_dst_cache_roi);
                }
            }
        }
    }
};

}
