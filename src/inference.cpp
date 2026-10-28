/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */
#include <cctype>

#include "inference.h"
#include "output_size.h"
#include "utils.h"

#include <limits>

using namespace std;
namespace triton {
namespace backend {
namespace npu_ge {

#define TYPE_32_BYTE 4
#define TYPE_64_BYTE 8
#define TYPE_16_BYTE 2
#define TYPE_8_BYTE 1
#define BYTE_PTR char *

#define PRIORITY_HIGH 2
#define PRIORITY_MID 1
#define PRIORITY_LOW 0
#define OPERATOR_PAIR 2

// 初始化类型映射
void Inference::InitTypeMappings()
{
    size_by_type_ = {{TRITONSERVER_TYPE_BOOL, TYPE_8_BYTE},    {TRITONSERVER_TYPE_UINT8, TYPE_8_BYTE},
                     {TRITONSERVER_TYPE_UINT16, TYPE_16_BYTE}, {TRITONSERVER_TYPE_UINT32, TYPE_32_BYTE},
                     {TRITONSERVER_TYPE_UINT64, TYPE_64_BYTE}, {TRITONSERVER_TYPE_INT8, TYPE_8_BYTE},
                     {TRITONSERVER_TYPE_INT16, TYPE_16_BYTE},  {TRITONSERVER_TYPE_INT32, TYPE_32_BYTE},
                     {TRITONSERVER_TYPE_INT64, TYPE_64_BYTE},  {TRITONSERVER_TYPE_FP16, TYPE_16_BYTE},
                     {TRITONSERVER_TYPE_FP32, TYPE_32_BYTE},   {TRITONSERVER_TYPE_FP64, TYPE_64_BYTE},
                     {TRITONSERVER_TYPE_BYTES, TYPE_8_BYTE},   {TRITONSERVER_TYPE_BF16, TYPE_16_BYTE}};

    ge_data_trans_ = {{TRITONSERVER_TYPE_BOOL, ge::DT_BOOL},     {TRITONSERVER_TYPE_UINT8, ge::DT_UINT8},
                      {TRITONSERVER_TYPE_UINT16, ge::DT_UINT16}, {TRITONSERVER_TYPE_UINT32, ge::DT_UINT32},
                      {TRITONSERVER_TYPE_UINT64, ge::DT_UINT64}, {TRITONSERVER_TYPE_INT8, ge::DT_INT8},
                      {TRITONSERVER_TYPE_INT16, ge::DT_INT16},   {TRITONSERVER_TYPE_INT32, ge::DT_INT32},
                      {TRITONSERVER_TYPE_INT64, ge::DT_INT64},   {TRITONSERVER_TYPE_FP16, ge::DT_FLOAT16},
                      {TRITONSERVER_TYPE_FP32, ge::DT_FLOAT},    {TRITONSERVER_TYPE_BYTES, ge::DT_STRING},
                      {TRITONSERVER_TYPE_BF16, ge::DT_BF16}};
}

void Inference::PrintModelTensorInfo(const std::vector<ModelTensorInfo> &tensor_info)
{
    for (size_t j = 0; j < tensor_info.size(); j++) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("name ") + tensor_info[j].name_).c_str());
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                    (std::string("dtype_ ") + to_string(static_cast<int>(tensor_info[j].dtype_))).c_str());
        for (size_t i = 0; i < tensor_info[j].tensor_dims_.size(); i++) {
            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("dim") + to_string(static_cast<int>(i)) + " " +
                                                   to_string(static_cast<int>(tensor_info[j].tensor_dims_[i])))
                                                      .c_str());
        }
    }
}

void Inference::SetTensorInfo(ModelState *model_state_)
{
    input_tensor_info_.clear();
    output_tensor_info_.clear();
    vector<int64_t> &preferBatch = model_state_->GetDynamicmode().preferred_batch_sizes;
    int input_num = model_state_->GetInputCount();
    int output_num = model_state_->GetOutputCount();
    for (int i = 0; i < input_num; i++) {
        auto tensor = model_state_->GetInputClientTensors()[i];
        ModelTensorInfo modelTensorInfo;
        modelTensorInfo.name_ = tensor.name_;
        modelTensorInfo.dtype_ = tensor.dtype_;
        if (preferBatch.size() > 0 && (model_state_->GetModelNode() == ModelState::ModelMode::MAX_BATCH)) {
            modelTensorInfo.tensor_dims_ = tensor.dims_;
        }
        input_tensor_info_.push_back(modelTensorInfo);
    }
    for (int i = 0; i < output_num; i++) {
        auto tensor = model_state_->GetOutputClientTensors()[i];
        ModelTensorInfo modelTensorInfo;
        modelTensorInfo.name_ = tensor.name_;
        modelTensorInfo.dtype_ = tensor.dtype_;
        output_tensor_info_.push_back(modelTensorInfo);
    }
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("ModelTensorInfo: ")).c_str());
    PrintModelTensorInfo(input_tensor_info_);
    PrintModelTensorInfo(output_tensor_info_);
}

int Inference::GetTotalBatch(const std::vector<int> &batch_result)
{
    int batch_total = 0;
    for (auto &batch : batch_result) {
        batch_total += batch;
    }
    return batch_total;
}

Inference::Inference(ModelState *model_state, TRITONBACKEND_ModelInstance *model_instance)
{
    InitTypeMappings();
    model_state_ = model_state;
    model_instance_ = model_instance;
    SetTensorInfo(model_state_);
}

int Inference::GetValueByKey(size_t key) const
{
    auto it = size_by_type_.find(key);
    if (it != size_by_type_.end()) {
        return it->second;
    } else {
        std::cerr << "键 " << key << " 不存在" << std::endl;
        return RET_ERR;
    }
}

int Inference::Start()
{
    if (model_state_ == nullptr) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("modelState_ == nullptr Start fail")).c_str());
        return RET_ERR;
    }
    if (status_ == Scheduler::Status::RUNNING) {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("Dynamicmode param already set up")).c_str());
        return RET_OK;
    }

    int64_t delaytime = model_state_->GetDynamicmode().max_queue_delay_microseconds_;
    timeout_ = std::chrono::microseconds(delaytime);

    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("timeout_ ") + std::to_string(delaytime)).c_str());

    vector<int64_t> &preferBatch = model_state_->GetDynamicmode().preferred_batch_sizes;
    for (size_t i = 0; i < preferBatch.size(); i++) {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO,
                    (std::string("preferBatch") + std::to_string(i) + " " + std::to_string(preferBatch[i])).c_str());
    }

    status_ = Scheduler::Status::RUNNING;
    return RET_OK;
}

// 组合批次后计算分配
void Inference::CalculateCombineBatchDistribution(std::vector<int> &input_offset,
                                                  const std::vector<int> &batch_combine)
{
    input_offset.push_back(0);
    for (size_t i = 1; i < batch_combine.size(); ++i) {
        int temp = input_offset.back() + batch_combine[i - 1];
        input_offset.push_back(temp);
    }
}

bool Inference::CanCombine(ModelState *model_state)
{
    return model_state->GetDynamicmode().preferred_batch_sizes.size() > 0 &&
           model_state->GetModelNode() == ModelState::ModelMode::MAX_BATCH && model_state->GetInToOutMap().size() == 0;
}

// 获取执行批次参数
void Inference::GetExecutionBatchParams(int batch, int &exec_batch, int &cycle_count)
{
    if (model_state_->GetGeStaticMode()) {
        if (model_state_->GetInferMode() == ModelState::InferMode::DYNAMICMODEL ||
            model_state_->GetDynamicBatchSupport()) {
            exec_batch = 1;
            cycle_count = batch;
        } else {
            exec_batch = batch;
            cycle_count = 1;
        }
    } else {
        exec_batch = batch;
        cycle_count = 1;
    }
}

ge::DataType Inference::TransToGeData(size_t key) const
{
    auto it = ge_data_trans_.find(key);
    if (it != ge_data_trans_.end()) {
        return it->second;
    } else {
        std::cerr << "键 " << key << " 不存在" << std::endl;
        return ge::DT_MAX;
    }
}

void Inference::FreeDevResources(std::vector<void *> &dev_buffer_)
{
    for (size_t i = 0; i < dev_buffer_.size(); i++) {
        aclrtFree(dev_buffer_[i]);
    }
    dev_buffer_.clear();
}

// 等待所有线程完成
void Inference::JoinAllThreads(std::vector<std::thread> &threads)
{
    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

// 计算批次分配
void Inference::CalculateBatchDistribution(int batch_total, size_t instance_count, std::vector<int> &instance_offset,
                                           std::vector<int> &batch_result)
{
    instance_offset.resize(instance_count, 0);
    batch_result.resize(instance_count, 0);

    size_t base = batch_total / instance_count;
    size_t remainder = batch_total % instance_count;

    for (size_t i = 0; i < instance_count; ++i) {
        batch_result[i] = base;
        if (i < remainder) {
            batch_result[i] += 1;
        }
    }

    for (size_t i = 1; i < instance_count; ++i) {
        instance_offset[i] = instance_offset[i - 1] + batch_result[i - 1];
    }
}

int Inference::AllocateSingleMemoryV2(ModelState *model_state, size_t j, std::vector<void *> &indev_buffer_,
                                      std::vector<int> &indev_line_size_, const int64_t *shape, uint32_t dims_count,
                                      uint64_t buffer_size)
{
    // 申请dev侧内存，输入数据由调用方按buffer片段顺序复制。
    aclError acl_ret;
    int element_size = GetValueByKey(model_state->GetInputClientTensors()[j].dtype_);
    if (element_size <= 0 || (dims_count > 0 && shape == nullptr) ||
        buffer_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return RET_ERR;
    }

    uint64_t per_buffer_size = static_cast<uint64_t>(element_size);
    vector<int64_t> v1;
    for (size_t i = 0; i < dims_count; i++) {
        if (shape[i] < 0) {
            return RET_ERR;
        }
        v1.push_back(shape[i]);
    }
    input_tensor_info_[j].tensor_dims_ = v1;

    int index = 1;
    ModelState::ModelMode modelmode = model_state_->GetModelNode();
    if (modelmode == ModelState::ModelMode::NO_MAX_BATCH_FIRST_NOT_SAME ||
        modelmode == ModelState::ModelMode::NO_MAX_BATCH_FIRST_NOT_SAME_EXIST_NEGATIVE) {
        index = 0;
    }
    for (size_t k = index; k < dims_count; k++) {
        if (shape[k] != 0 && per_buffer_size > std::numeric_limits<uint64_t>::max() /
                                                    static_cast<uint64_t>(shape[k])) {
            return RET_ERR;
        }
        per_buffer_size *= static_cast<uint64_t>(shape[k]);
    }
    if (per_buffer_size > static_cast<uint64_t>(INT_MAX)) {
        return RET_ERR;
    }

    void *indev_buffer = nullptr;
    acl_ret = aclrtMalloc(&indev_buffer, static_cast<size_t>(buffer_size), ACL_MEM_MALLOC_NORMAL_ONLY);
    if (acl_ret != ACL_ERROR_NONE) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                    (std::string("malloc dev buffer failed, ret is:") + std::to_string(acl_ret)).c_str());
        return RET_ERR;
    }
    indev_buffer_.push_back(indev_buffer);
    indev_line_size_.push_back(static_cast<int>(per_buffer_size));
    return RET_OK;
}

int Inference::ProcessRequestInputsV2(TRITONBACKEND_Request *request, size_t request_index,
                                      const std::vector<int> &batch_result, std::vector<void *> &indev_buffer_,
                                      std::vector<int> &indev_line_size_, const std::vector<int> &input_offset)
{  // 得到每个张量 每行的大小 默认维度0是批次
    aclError acl_ret;
    for (size_t j = 0; j < model_state_->GetInputCount(); j++) {
        TRITONBACKEND_Input *input;
        TRITONBACKEND_RequestInput(request, model_state_->GetInputClientTensors()[j].name_.c_str(), &input);
        TRITONSERVER_DataType datatype;
        const int64_t *shape;
        uint32_t dims_count;
        const char *input_name;
        uint64_t reported_byte_size;
        uint32_t buffer_count;
        TRITONSERVER_Error *error = TRITONBACKEND_InputProperties(input, &input_name, &datatype, &shape, &dims_count,
                                                                  &reported_byte_size, &buffer_count);
        if (error != nullptr) {
            TRITONSERVER_ErrorDelete(error);
            return RET_ERR;
        }
        if (datatype != static_cast<TRITONSERVER_DataType>(model_state_->GetInputClientTensors()[j].dtype_) ||
            (dims_count > 0 && shape == nullptr)) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "input datatype or shape does not match model metadata");
            return RET_ERR;
        }

        int element_size = GetValueByKey(datatype);
        if (element_size <= 0) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "unsupported input datatype");
            return RET_ERR;
        }
        uint64_t expected_byte_size = reported_byte_size;
        if (datatype != TRITONSERVER_TYPE_BYTES) {
            expected_byte_size = static_cast<uint64_t>(element_size);
            for (uint32_t dim = 0; dim < dims_count; ++dim) {
                if (shape[dim] < 0 ||
                    (shape[dim] != 0 && expected_byte_size > std::numeric_limits<uint64_t>::max() /
                                                                   static_cast<uint64_t>(shape[dim]))) {
                    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "input shape byte size overflows");
                    return RET_ERR;
                }
                expected_byte_size *= static_cast<uint64_t>(shape[dim]);
            }
        }

        struct InputBufferSegment {
            const void *buffer;
            uint64_t byte_size;
            TRITONSERVER_MemoryType memory_type;
            int64_t memory_type_id;
        };
        vector<InputBufferSegment> input_buffers;
        input_buffers.reserve(buffer_count);
        uint64_t total_buffer_size = 0;
        for (uint32_t buffer_index = 0; buffer_index < buffer_count; ++buffer_index) {
            InputBufferSegment segment{};
            segment.memory_type = TRITONSERVER_MEMORY_CPU;
            segment.memory_type_id = 0;
            error = TRITONBACKEND_InputBuffer(input, buffer_index, &segment.buffer, &segment.byte_size,
                                              &segment.memory_type, &segment.memory_type_id);
            if (error != nullptr) {
                TRITONSERVER_ErrorDelete(error);
                return RET_ERR;
            }
            if ((segment.byte_size > 0 && segment.buffer == nullptr) ||
                segment.byte_size > expected_byte_size - total_buffer_size) {
                LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "input buffer fragments exceed the tensor byte size");
                return RET_ERR;
            }
            if (segment.memory_type != TRITONSERVER_MEMORY_CPU &&
                segment.memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
                LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "input buffer memory type is not host accessible");
                return RET_ERR;
            }
            total_buffer_size += segment.byte_size;
            input_buffers.push_back(segment);
        }
        if (total_buffer_size != reported_byte_size || total_buffer_size != expected_byte_size) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "input buffer size does not match tensor shape and datatype");
            return RET_ERR;
        }

        void *destination = nullptr;
        if (CanCombine(model_state_)) {
            if (j >= indev_buffer_.size() || j >= indev_line_size_.size() ||
                request_index >= batch_result.size() || request_index >= input_offset.size() ||
                indev_line_size_[j] <= 0 || batch_result[request_index] < 0 || input_offset[request_index] < 0) {
                LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                            "input shape does not match the combined batch slice");
                return RET_ERR;
            }
            uint64_t line_size = static_cast<uint64_t>(indev_line_size_[j]);
            if (static_cast<uint64_t>(batch_result[request_index]) >
                    std::numeric_limits<uint64_t>::max() / line_size ||
                static_cast<uint64_t>(batch_result[request_index]) * line_size != expected_byte_size) {
                LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "input shape does not match the combined batch slice");
                return RET_ERR;
            }
            uint64_t total_batch = 0;
            for (int batch_size : batch_result) {
                if (batch_size < 0 || static_cast<uint64_t>(batch_size) >
                                          std::numeric_limits<uint64_t>::max() - total_batch) {
                    return RET_ERR;
                }
                total_batch += static_cast<uint64_t>(batch_size);
            }
            if (total_batch > std::numeric_limits<uint64_t>::max() / line_size ||
                static_cast<uint64_t>(input_offset[request_index]) >
                    std::numeric_limits<uint64_t>::max() / line_size) {
                return RET_ERR;
            }
            uint64_t capacity = total_batch * line_size;
            uint64_t offset = static_cast<uint64_t>(input_offset[request_index]) * line_size;
            if (indev_buffer_[j] == nullptr || offset > capacity || expected_byte_size > capacity - offset) {
                LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "combined input slice exceeds the allocated buffer");
                return RET_ERR;
            }
            destination = static_cast<char *>(indev_buffer_[j]) + offset;
        } else {
            CHECK_RET_ERR(AllocateSingleMemoryV2(model_state_, j, indev_buffer_, indev_line_size_, shape, dims_count,
                                                 expected_byte_size));
            destination = indev_buffer_.back();
        }

        uint64_t copied_size = 0;
        for (const auto &segment : input_buffers) {
            if (segment.byte_size == 0) {
                continue;
            }
            if (segment.memory_type != TRITONSERVER_MEMORY_CPU &&
                segment.memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
                LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "input buffer memory type is not host accessible");
                return RET_ERR;
            }
            acl_ret = aclrtMemcpy(static_cast<char *>(destination) + copied_size,
                                  static_cast<size_t>(expected_byte_size - copied_size), segment.buffer,
                                  static_cast<size_t>(segment.byte_size), ACL_MEMCPY_HOST_TO_DEVICE);
            if (acl_ret != ACL_SUCCESS) {
                LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                            (std::string("aclrtMemcpy is failed, ret code is ") + std::to_string(acl_ret)).c_str());
                return RET_ERR;
            }
            copied_size += segment.byte_size;
        }
    }
    return RET_OK;
}

int Inference::ProcessInputBuffersV2(const std::vector<TRITONBACKEND_Request *> &batch_tasks,
                                     const std::vector<int> &batch_result, std::vector<int> &input_offset,
                                     std::vector<void *> &indev_buffer_, std::vector<int> &indev_line_size_)
{
    for (size_t i = 0; i < batch_tasks.size(); i++) {
        TRITONBACKEND_Request *request = batch_tasks[i];
        CHECK_RET_ERR(ProcessRequestInputsV2(request, i, batch_result, indev_buffer_, indev_line_size_, input_offset));
    }
    return RET_OK;
}

int Inference::AllocateCombinedMemoryV2(ModelState *model_state, std::vector<void *> &indev_buffer_,
                                        std::vector<int> &indev_line_size_, int batch_total)
{
    aclError acl_ret;
    for (size_t i = 0; i < model_state->GetInputCount(); i++) {
        size_t sample_element_size = GetValueByKey(model_state->GetInputClientTensors()[i].dtype_);  // 单个元素大小
        int64_t total_out_size = static_cast<int64_t>(sample_element_size);
        int per_buffer_size = sample_element_size;
        if (input_tensor_info_[i].tensor_dims_.size() == 0) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                        (std::string("input_tensor_info_[i].tensor_dims_.size() is: empty")).c_str());
            return RET_ERR;
        }
        vector<int64_t> &dims_in = input_tensor_info_[i].tensor_dims_;
        total_out_size *= batch_total;
        for (size_t k = 0; k < dims_in.size(); k++) {
            total_out_size *= dims_in[k];
            per_buffer_size *= dims_in[k];
        }
        void *indev_buffer = nullptr;
        acl_ret = aclrtMalloc(&indev_buffer, total_out_size, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (acl_ret != ACL_ERROR_NONE) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                        (std::string("malloc dev buffer failed, ret is:") + std::to_string(acl_ret)).c_str());
            return RET_ERR;
        }
        indev_buffer_.push_back(indev_buffer);
        indev_line_size_.push_back(per_buffer_size);
    }
    return RET_OK;
}

int Inference::ExtractCombineInputInfoV2(int batch_total, const std::vector<TRITONBACKEND_Request *> &batch_tasks,
                                         const std::vector<int> &batch_result, std::vector<int> &input_offset,
                                         std::vector<void *> &indev_buffer_, std::vector<int> &indev_line_size_)
{
    // 计算每个请求的起始复制位置
    CalculateCombineBatchDistribution(input_offset, batch_result);
    if (CanCombine(model_state_)) {
        CHECK_RET_ERR(AllocateCombinedMemoryV2(model_state_, indev_buffer_, indev_line_size_, batch_total));
    }
    return ProcessInputBuffersV2(batch_tasks, batch_result, input_offset, indev_buffer_, indev_line_size_);
}

// 准备输出缓冲区
int Inference::PrepareOutputBuffersV2(std::vector<void *> &outhost_buffer_, std::vector<int64_t> &outsize,
                                      std::vector<int64_t> &outhost_line_size_, int batch_total)
{
    aclError acl_ret;

    for (size_t j = 0; j < model_state_->GetOutputCount(); j++) {
        size_t sample_element_size = GetValueByKey(model_state_->GetOutputClientTensors()[j].dtype_);
        vector<int64_t> dims_out = output_tensor_info_[j].tensor_dims_;
        const bool include_batch = model_state_->GetInferMode() == ModelState::InferMode::DYNAMICMODEL ||
                                   model_state_->GetDynamicBatchSupport();
        if (include_batch) {
            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("use batch_total: ") + to_string(batch_total)).c_str());
        }
        int index = 0;
        ModelState::ModelMode modelmode = model_state_->GetModelNode();
        if (modelmode == ModelState::ModelMode::NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE ||
            modelmode == ModelState::ModelMode::NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE_HAVE_UNKNOW_DIM) {
            index = 1;
        }
        int64_t total_out_size = 0;
        int64_t per_buffer_size = 0;
        if (!CalculateOutputByteSizes(sample_element_size, dims_out, include_batch, batch_total,
                                      static_cast<size_t>(index), total_out_size, per_buffer_size)) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "output byte size is invalid or exceeds int64 range");
            return RET_ERR;
        }
        for (auto dim : dims_out) {
            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("out dim: ") + to_string(dim)).c_str());
        }

        void *outhost_buffer = nullptr;
        acl_ret = aclrtMallocHost(&outhost_buffer, total_out_size);
        if (acl_ret != ACL_SUCCESS) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                        (std::string("malloc dev buffer failed, ret is:") + std::to_string(acl_ret)).c_str());
            return RET_ERR;
        }
        outhost_buffer_.push_back(outhost_buffer);
        outsize.push_back(total_out_size);
        outhost_line_size_.push_back(per_buffer_size);
    }
    return RET_OK;
}

void Inference::FreeHostResourcesV2(std::vector<void *> &outhost_buffer_)
{
    for (size_t i = 0; i < outhost_buffer_.size(); i++) {
        aclrtFreeHost(outhost_buffer_[i]);
    }
    outhost_buffer_.clear();
}

// 构建输入张量
int Inference::BuildInputTensorV2(size_t input_index, int exec_batch, std::vector<void *> &indev_buffer_,
                                  gert::Tensor &tensor, int cycle_count, int instance_offset,
                                  std::vector<int> &indev_line_size_, std::vector<void *> &oneBtachDev)
{
    // TODO 如果循环次数是1 直接使用偏移的数据地址 如果循环次数是exec_batch 重新申请一块内存 再添加一次复制操作 dev->dev
    std::initializer_list<int64_t> origin_shape_list;
    std::initializer_list<int64_t> storage_shape_list;
    gert::StorageShape input_shape(origin_shape_list, storage_shape_list);
    gert::Shape &origin_shape = input_shape.MutableOriginShape();
    gert::Shape &storage_shape = input_shape.MutableStorageShape();

    if (model_state_->GetInferMode() == ModelState::InferMode::DYNAMICMODEL || model_state_->GetDynamicBatchSupport()) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                    (std::string("input exec_batch append dim is ") + std::to_string(exec_batch)).c_str());
        origin_shape.AppendDim(exec_batch);
        storage_shape.AppendDim(exec_batch);
    }
    vector<int64_t> dims_in;
    int index = 1;
    ModelState::ModelMode modelmode = model_state_->GetModelNode();
    if (modelmode == ModelState::ModelMode::NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE ||
        modelmode == ModelState::ModelMode::NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE_HAVE_UNKNOW_DIM ||
        modelmode == ModelState::ModelMode::NO_MAX_BATCH_FIRST_NOT_SAME ||
        modelmode == ModelState::ModelMode::NO_MAX_BATCH_FIRST_NOT_SAME_EXIST_NEGATIVE) {
        index = 0;
    }
    if (model_state_->GetModelNode() == ModelState::ModelMode::MAX_BATCH && model_state_->GetInToOutMap().size() == 0) {
        index = 1;
    }
    vector<int64_t> &preferBatch = model_state_->GetDynamicmode().preferred_batch_sizes;
    if (preferBatch.size() > 0 &&
        (model_state_->GetModelNode() ==
         ModelState::ModelMode::MAX_BATCH)) {  // 批处理情况下 输入维度从配置文件设置 非批处理 从实际接受张量维度设置
        index = 0;
    }
    for (size_t i = index; i < input_tensor_info_[input_index].tensor_dims_.size(); i++) {
        dims_in.push_back(input_tensor_info_[input_index].tensor_dims_[i]);
    }
    for (size_t i = 0; i < dims_in.size(); i++) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                    (std::string("input append dim is ") + std::to_string(dims_in[i])).c_str());
        origin_shape.AppendDim(dims_in[i]);
        storage_shape.AppendDim(dims_in[i]);
    }

    ge::DataType input_dtype = TransToGeData(model_state_->GetInputClientTensors()[input_index].dtype_);
    if (input_dtype == ge::DT_MAX) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("TransToGeData failed, ret is:") +
                                             std::to_string(model_state_->GetInputClientTensors()[input_index].dtype_))
                                                .c_str());
        return RET_ERR;
    }

    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                (std::string("set input_dtype is ") + std::to_string(static_cast<int>(input_dtype))).c_str());

    gert::StorageFormat input_format(ge::FORMAT_ND, ge::FORMAT_ND, {});
    void *dev_buffer;
    if (exec_batch == 1 && cycle_count > 1) {  // 走one batch分支
        aclError acl_ret;
        int buffer_size = exec_batch * indev_line_size_[input_index];
        acl_ret = aclrtMalloc(&dev_buffer, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (acl_ret != ACL_ERROR_NONE) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                        (std::string("malloc dev buffer failed, ret is:") + std::to_string(acl_ret)).c_str());
            return RET_ERR;
        }
        oneBtachDev.push_back(dev_buffer);
    } else {
        dev_buffer = (void *)((BYTE_PTR)indev_buffer_[input_index] + instance_offset * indev_line_size_[input_index]);
    }
    tensor = {input_shape, input_format, gert::kOnDeviceHbm, input_dtype, dev_buffer};

    return RET_OK;
}

// 准备输入张量
int Inference::PrepareInputTensorsV2(std::vector<int> &indev_line_size_, int exec_batch, int instance_index,
                                     int instance_offset, std::vector<void *> &indev_buffer_,
                                     std::vector<gert::Tensor> &inputs, int cycle_count,
                                     std::vector<void *> &oneBtachDev)
{
    inputs.resize(model_state_->GetInputCount());

    for (size_t j = 0; j < model_state_->GetInputCount(); j++) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("current instance_index") + std::to_string(instance_index) +
                                               " : " + std::to_string(exec_batch))
                                                  .c_str());

        // 构建输入张量
        if (BuildInputTensorV2(j, exec_batch, indev_buffer_, inputs[j], cycle_count, instance_offset,
                               indev_line_size_, oneBtachDev) != RET_OK) {
            return RET_ERR;
        }
    }
    return RET_OK;
}

// one batch 场景下按 cycle 把本实例分片从 dev 侧复制到输入张量的 dev 缓冲
int Inference::CopyOneBatchInputsV2(std::vector<void *> &indev_buffer_, std::vector<int> &indev_line_size_,
                                    std::vector<gert::Tensor> &inputs, int exec_batch, int offset)
{
    for (size_t j = 0; j < model_state_->GetInputCount(); j++) {
        int buffer_size = exec_batch * indev_line_size_[j];
        void *src_ptr = (void *)((BYTE_PTR)indev_buffer_[j] + offset * indev_line_size_[j]);
        aclError acl_ret =
            aclrtMemcpy(inputs[j].GetAddr(), buffer_size, src_ptr, buffer_size, ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (acl_ret != ACL_SUCCESS) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                        (std::string("aclrtMemcpy is failed, ret code is ") + std::to_string(acl_ret)).c_str());
            return RET_ERR;
        }
    }
    return RET_OK;
}

// 执行一次图并等待流同步
int Inference::ExecuteGraphAndSyncV2(Scheduler::Instance *instance, std::vector<gert::Tensor> &inputs,
                                     std::vector<gert::Tensor> &outputs)
{
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "start infer!: ");
    ge::Status ret = instance->session->ExecuteGraphWithStreamAsync(instance->index, instance->stream, inputs, outputs);
    if (ret != RET_OK) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                    (std::string("execute model failed, ret is: ") + std::to_string(ret)).c_str());
        return RET_ERR;
    }
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("aclrtSynchronizeStream: ")).c_str());
    aclError acl_ret = aclrtSynchronizeStream(instance->stream);
    if (acl_ret != ACL_SUCCESS) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                    (std::string("aclrtSynchronizeStream is failed, ret code is ") + std::to_string(acl_ret)).c_str());
        return RET_ERR;
    }
    // 记录本实例的“计算结束”时刻：此时 GE 执行已完成，输出回拷尚未开始
    SET_TIMESTAMP(instance->compute_end_ns);
    return RET_OK;
}

// 把本次周期的输出从 dev 侧复制回主机缓冲的本实例分片
int Inference::CopyOutputsToHostV2(std::vector<void *> &outhost_buffer_, std::vector<int64_t> &outhost_line_size_,
                                   std::vector<gert::Tensor> &outputs, int exec_batch, int offset)
{
    for (size_t j = 0; j < model_state_->GetOutputCount(); j++) {
        int64_t total_out_size = 0;
        if (!CheckedMultiplyOutputSize(outhost_line_size_[j], exec_batch, total_out_size)) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "output copy size exceeds int64 range");
            return RET_ERR;
        }
        void *dst_ptr = (void *)((BYTE_PTR)outhost_buffer_[j] + offset * outhost_line_size_[j]);
        aclError acl_ret =
            aclrtMemcpy(dst_ptr, total_out_size, outputs[j].GetAddr(), total_out_size, ACL_MEMCPY_DEVICE_TO_HOST);
        if (acl_ret != ACL_SUCCESS) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                        (std::string("aclrtMemcpy output is failed, ret code is ") + std::to_string(acl_ret)).c_str());
            return RET_ERR;
        }
    }
    return RET_OK;
}

int Inference::ExecuteInferenceCycleV2(Scheduler::Instance *instance, std::vector<void *> &indev_buffer_,
                                       std::vector<int> &indev_line_size_, std::vector<void *> &outhost_buffer_,
                                       std::vector<int64_t> &outhost_line_size_, std::vector<gert::Tensor> &inputs,
                                       std::vector<gert::Tensor> &outputs, int exec_batch, int cycle_count,
                                       int instance_index, const std::vector<int> &instance_offset)
{
    for (int k = 0; k < cycle_count; k++) {
        int offset = instance_offset[instance_index] + k;
        // one batch 循环 cycle_count 次，从 dev 侧复制数据到 dev 侧
        if (cycle_count > 1 && exec_batch == 1 &&
            CopyOneBatchInputsV2(indev_buffer_, indev_line_size_, inputs, exec_batch, offset) != RET_OK) {
            return RET_ERR;
        }
        if (ExecuteGraphAndSyncV2(instance, inputs, outputs) != RET_OK) {
            return RET_ERR;
        }
        if (CopyOutputsToHostV2(outhost_buffer_, outhost_line_size_, outputs, exec_batch, offset) != RET_OK) {
            return RET_ERR;
        }
    }
    return RET_OK;
}

// 准备输出张量
int Inference::PrepareOutputTensors(std::vector<int64_t> &outhost_line_size_, int exec_batch,
                                    std::vector<void *> &outdev_buffer_, std::vector<gert::Tensor> &outputs)
{
    aclError acl_ret;
    outputs.resize(model_state_->GetOutputCount());

    for (size_t j = 0; j < model_state_->GetOutputCount(); j++) {
        int64_t total_out_size = 0;
        if (!CheckedMultiplyOutputSize(outhost_line_size_[j], exec_batch, total_out_size)) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "output allocation size exceeds int64 range");
            return RET_ERR;
        }
        void *out_dev_buffer = nullptr;
        acl_ret = aclrtMalloc(&out_dev_buffer, total_out_size, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (acl_ret != ACL_ERROR_NONE) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                        (std::string("malloc host buffer failed, ret is:") + std::to_string(static_cast<int>(acl_ret)))
                            .c_str());
            return RET_ERR;
        }
        outdev_buffer_.push_back(out_dev_buffer);

        // 构建输出张量
        if (BuildOutputTensor(j, exec_batch, out_dev_buffer, outputs[j]) != RET_OK) {
            return RET_ERR;
        }
    }
    return RET_OK;
}

// 构建输出张量
int Inference::BuildOutputTensor(size_t output_index, int exec_batch, void *out_dev_buffer, gert::Tensor &tensor)
{
    std::initializer_list<int64_t> origin_shape_list_out;
    std::initializer_list<int64_t> storage_shape_listt_out;
    gert::StorageShape output_shape(origin_shape_list_out, storage_shape_listt_out);

    gert::Shape &origin_shape_out = output_shape.MutableOriginShape();
    gert::Shape &storage_shape_out = output_shape.MutableStorageShape();

    if (model_state_->GetInferMode() == ModelState::InferMode::DYNAMICMODEL || model_state_->GetDynamicBatchSupport()) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                    (std::string("output exec_batch append dim is ") + std::to_string(exec_batch)).c_str());
        origin_shape_out.AppendDim(exec_batch);
        storage_shape_out.AppendDim(exec_batch);
    }

    vector<int64_t> dims_out = output_tensor_info_[output_index].tensor_dims_;
    for (auto dim : dims_out) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("output append dim is ") + std::to_string(dim)).c_str());
        origin_shape_out.AppendDim(dim);
        storage_shape_out.AppendDim(dim);
    }

    ge::DataType output_dtype = TransToGeData(model_state_->GetOutputClientTensors()[output_index].dtype_);
    if (output_dtype == ge::DT_MAX) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("TransToGeData failed, ret is:") + std::to_string(static_cast<int>(output_dtype))).c_str());
        return RET_ERR;
    }

    gert::StorageFormat output_format(ge::FORMAT_ND, ge::FORMAT_ND, {});
    tensor = {output_shape, output_format, gert::kOnDeviceHbm, output_dtype, out_dev_buffer};

    return RET_OK;
}

// 执行单个推理实例
int Inference::ExecuteSingleInferenceV2(Scheduler::Instance *instance, int instance_index, int instance_count,
                                        std::vector<void *> &indev_buffer_, std::vector<int> &indev_line_size_,
                                        std::vector<void *> &outhost_buffer_, std::vector<int64_t> &outhost_line_size_,
                                        const std::vector<int> &instance_offset,
                                        const std::vector<int> &batch_result)

{
    int batch = batch_result[instance_index];
    if (batch == 0) {
        return RET_OK;  // 无需执行
    }

    int exec_batch;
    int cycle_count;
    GetExecutionBatchParams(batch, exec_batch, cycle_count);

    aclrtSetCurrentContext(instance->context);

    std::vector<void *> outdev_buffer_;
    std::vector<gert::Tensor> inputs;
    std::vector<gert::Tensor> outputs;
    std::vector<void *> oneBtachDev;

    // 准备输入张量
    if (PrepareInputTensorsV2(indev_line_size_, exec_batch, instance_index, instance_offset[instance_index],
                              indev_buffer_, inputs, cycle_count, oneBtachDev) != RET_OK) {
        FreeDevResources(outdev_buffer_);
        FreeDevResources(oneBtachDev);
        return RET_ERR;
    }

    // 准备输出张量
    if (PrepareOutputTensors(outhost_line_size_, exec_batch, outdev_buffer_, outputs) != RET_OK) {
        FreeDevResources(outdev_buffer_);
        FreeDevResources(oneBtachDev);
        return RET_ERR;
    }

    // 执行推理循环
    if (ExecuteInferenceCycleV2(instance, indev_buffer_, indev_line_size_, outhost_buffer_, outhost_line_size_, inputs,
                                outputs, exec_batch, cycle_count, instance_index, instance_offset) != RET_OK) {
        FreeDevResources(outdev_buffer_);
        FreeDevResources(oneBtachDev);
        return RET_ERR;
    }

    FreeDevResources(outdev_buffer_);
    FreeDevResources(oneBtachDev);
    return RET_OK;
}

// 处理单个推理实例的公共方法
int Inference::ProcessSingleInstanceV2(Scheduler::Instance *instance, int instance_index, int instance_count,
                                       std::vector<void *> &indev_buffer_, std::vector<int> &indev_line_size_,
                                       std::vector<void *> &outhost_buffer_, std::vector<int64_t> &outhost_line_size_,
                                       const std::vector<int> &instance_offset, const std::vector<int> &batch_result,

                                       int &success_count, int &invalid_count)
{
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("instance_index: ") + std::to_string(instance_index) +
                                           " :batch: " + std::to_string(batch_result[instance_index]))
                                              .c_str());

    int batch = batch_result[instance_index];
    if (batch == 0) {
        invalid_count++;
        return RET_OK;  // 无效实例，但不算错误
    }

    int result = ExecuteSingleInferenceV2(instance, instance_index, instance_count, indev_buffer_, indev_line_size_,
                                          outhost_buffer_, outhost_line_size_, instance_offset, batch_result);
    if (result == RET_OK) {
        success_count++;
    }

    return result;
}

// 串行执行推理（不使用线程）
int Inference::ExecuteInferenceV2(const std::vector<Scheduler::Instance *> &instances,
                                  std::vector<void *> &indev_buffer_, std::vector<int> &indev_line_size_,
                                  std::vector<void *> &outhost_buffer_, std::vector<int64_t> &outhost_line_size_,
                                  const std::vector<int> &instance_offset, const std::vector<int> &batch_result)
{
    if (instances.empty()) {
        return RET_OK;  // 没有实例，直接返回成功
    }

    int success_count = 0;
    int invalid_count = 0;

    for (size_t instance_index = 0; instance_index < instances.size(); instance_index++) {
        int ret = ProcessSingleInstanceV2(instances[instance_index], instance_index, instances.size(), indev_buffer_,
                                          indev_line_size_, outhost_buffer_, outhost_line_size_, instance_offset,
                                          batch_result, success_count, invalid_count);
        if (ret == RET_ERR) {
            return RET_ERR;
        }
    }

    return CheckInferenceResult(success_count, invalid_count, instances.size());
}

int Inference::ExecuteInferenceParallelV2(const std::vector<Scheduler::Instance *> &instances,
                                          std::vector<void *> &indev_buffer_, std::vector<int> &indev_line_size_,
                                          std::vector<void *> &outhost_buffer_,
                                          std::vector<int64_t> &outhost_line_size_,
                                          const std::vector<int> &instance_offset,
                                          const std::vector<int> &batch_result)
{
    if (instances.empty()) {
        return RET_OK;  // 没有实例，直接返回成功
    }

    std::vector<std::thread> threads;
    std::mutex mu;
    std::atomic<int> success_count{0};
    std::atomic<int> invalid_count{0};

    for (size_t instance_index = 0; instance_index < instances.size(); instance_index++) {
        threads.emplace_back([this, &mu, &success_count, &invalid_count, instance = instances[instance_index],
                              instance_index, count = instances.size(), &indev_buffer_, &indev_line_size_,
                              &outhost_buffer_, &outhost_line_size_, &instance_offset, &batch_result]() {
            // 使用局部变量来收集结果，然后原子更新
            int local_success = 0;
            int local_invalid = 0;

            ProcessSingleInstanceV2(instance, instance_index, count, indev_buffer_, indev_line_size_, outhost_buffer_,
                                    outhost_line_size_, instance_offset, batch_result, local_success, local_invalid);

            // 原子更新计数
            if (local_invalid > 0) {
                invalid_count += local_invalid;
            }
            if (local_success > 0) {
                success_count += local_success;
            }
        });
    }

    JoinAllThreads(threads);
    return CheckInferenceResult(success_count, invalid_count, instances.size());
}

// 检查推理结果的公共方法
int Inference::CheckInferenceResult(int success_count, int invalid_count, int total_instances)
{
    if (success_count != (total_instances - invalid_count) || invalid_count == total_instances) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "infer output failed!");
        return RET_ERR;
    }
    return RET_OK;
}

int Inference::SetInferenceContext(const std::vector<Scheduler::Instance *> &instances)
{
    aclrtContext nowContext;
    aclrtGetCurrentContext(&nowContext);
    if (nowContext != instances[0]->context) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "process request setcurrentContext ");
        aclrtSetCurrentContext(instances[0]->context);
    }
    return RET_OK;
}

int Inference::PrepareInferenceData(int batch_total, const std::vector<TRITONBACKEND_Request *> &batch_tasks,
                                    const std::vector<int> &batch_result, std::vector<int> &input_offset,
                                    std::vector<void *> &indev_buffer_, std::vector<int> &indev_line_size_,
                                    std::vector<void *> &outhost_buffer_, std::vector<int64_t> &outsize,
                                    std::vector<int64_t> &outhost_line_size_)
{
    // 提取输入信息
    if (ExtractCombineInputInfoV2(batch_total, batch_tasks, batch_result, input_offset, indev_buffer_,
                                  indev_line_size_) != RET_OK) {
        return RET_ERR;
    }

    // 计算未知维度
    CalcNegativeOne();

    // 准备输出缓冲区
    if (PrepareOutputBuffersV2(outhost_buffer_, outsize, outhost_line_size_, batch_total) != RET_OK) {
        FreeHostResourcesV2(outhost_buffer_);
        return RET_ERR;
    }

    return RET_OK;
}

int Inference::DispatchInference(const std::vector<Scheduler::Instance *> &instances,
                                 std::vector<void *> &indev_buffer_,
                                 std::vector<int> &indev_line_size_, std::vector<void *> &outhost_buffer_,
                                 std::vector<int64_t> &outhost_line_size_, const std::vector<int> &instance_offset,
                                 const std::vector<int> &instance_batch)
{
    if (instances.size() == 1) {
        if (ExecuteInferenceV2(instances, indev_buffer_, indev_line_size_, outhost_buffer_, outhost_line_size_,
                               instance_offset, instance_batch) == RET_ERR) {
            return RET_ERR;
        }
    } else {
        if (ExecuteInferenceParallelV2(instances, indev_buffer_, indev_line_size_, outhost_buffer_, outhost_line_size_,
                                       instance_offset, instance_batch) == RET_ERR) {
            return RET_ERR;
        }
    }
    return RET_OK;
}

int Inference::BuildCombineResponses(const std::vector<TRITONBACKEND_Request *> &batch_tasks,
                                     const std::vector<void *> &outhost_buffer_, const std::vector<int> &batch_result,
                                     const std::vector<int> &input_offset,
                                     const std::vector<int64_t> &outhost_line_size_,
                                     RequestTimestamps &timestamps)
{
    for (size_t i = 0; i < batch_tasks.size(); i++) {
        std::vector<int64_t> single_outsize_;
        for (size_t j = 0; j < model_state_->GetOutputCount(); j++) {
            int64_t response_size = 0;
            if (!CheckedMultiplyOutputSize(outhost_line_size_[j], batch_result[i], response_size)) {
                LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "response output size exceeds int64 range");
                return RET_ERR;
            }
            single_outsize_.push_back(response_size);
        }
        if (BuildComblineResponse(batch_tasks[i], outhost_buffer_, single_outsize_, batch_result[i], input_offset[i],
                                  outhost_line_size_, timestamps) != RET_OK) {
            return RET_ERR;
        }
    }

    // 整批统计：便于 Triton 计算 batch 维度的吞吐指标
    if (model_instance_ != nullptr) {
        LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportBatchStatistics(
            model_instance_, static_cast<size_t>(GetTotalBatch(batch_result)), timestamps.exec_start_ns,
            timestamps.compute_start_ns, timestamps.compute_end_ns, timestamps.exec_end_ns),
            "failed reporting batch statistics");
    }
    return RET_OK;
}

// 清空各实例上一轮采集到的计算结束时刻，避免复用旧值
void Inference::ResetInstanceComputeEnd(const std::vector<Scheduler::Instance *> &instances)
{
    for (auto *instance : instances) {
        instance->compute_end_ns = 0;
    }
}

// 取所有实例中最晚的 GE 执行结束时刻作为 compute_end（多实例/多 cycle 聚合）。
// 与 onnxruntime 后端一致：compute_end 停在 GE 执行完成处，输出回拷计入 compute_output。
void Inference::AggregateComputeEnd(const std::vector<Scheduler::Instance *> &instances, RequestTimestamps &timestamps)
{
    for (const auto *instance : instances) {
        if (instance->compute_end_ns > timestamps.compute_end_ns) {
            timestamps.compute_end_ns = instance->compute_end_ns;
        }
    }
    if (timestamps.compute_end_ns == 0) {
        // 兜底：没有采集到实例时间戳时退回当前时刻
        SET_TIMESTAMP(timestamps.compute_end_ns);
    }
}

int Inference::ProcessCombineRequestV2(const std::vector<TRITONBACKEND_Request *> &batch_tasks,
                                       const std::vector<Scheduler::Instance *> &instances,
                                       const std::vector<int> &batch_result)
{
    // 设置上下文
    if (SetInferenceContext(instances) != RET_OK) {
        return RET_ERR;
    }

    // 准备变量
    std::vector<int64_t> outsize;
    std::vector<void *> outhost_buffer_;
    std::vector<void *> indev_buffer_;
    std::vector<int> indev_line_size_;
    std::vector<int64_t> outhost_line_size_;
    std::vector<int> input_offset;
    int batch_total = GetTotalBatch(batch_result);
    // 统计时间点：本次 execute 的起点
    RequestTimestamps timestamps;
    SET_TIMESTAMP(timestamps.exec_start_ns);
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("batch_total: ") + std::to_string(batch_total)).c_str());

    // 准备推理数据（输入提取、维度计算、输出缓冲区准备）
    if (PrepareInferenceData(batch_total, batch_tasks, batch_result, input_offset,
                            indev_buffer_, indev_line_size_, outhost_buffer_, outsize,
                            outhost_line_size_) != RET_OK) {
        FreeDevResources(indev_buffer_);
        FreeHostResourcesV2(outhost_buffer_);
        return RET_ERR;
    }

    // 计算批次分布
    std::vector<int> instance_offset;
    std::vector<int> instance_batch;
    int instance_count = instances.size();
    CalculateBatchDistribution(batch_total, instance_count, instance_offset, instance_batch);

    // 执行推理
    ResetInstanceComputeEnd(instances);
    SET_TIMESTAMP(timestamps.compute_start_ns);
    int dispatch_result = DispatchInference(instances, indev_buffer_, indev_line_size_, outhost_buffer_,
                                           outhost_line_size_, instance_offset, instance_batch);
    AggregateComputeEnd(instances, timestamps);
    FreeDevResources(indev_buffer_);
    if (dispatch_result != RET_OK) {
        FreeHostResourcesV2(outhost_buffer_);
        return RET_ERR;
    }

    // 构建响应；exec_end 在 BuildComblineResponse 里写完全部响应输出后再刷新，
    // 这里先给一个兜底值，避免没有请求被处理时该字段为 0
    SET_TIMESTAMP(timestamps.exec_end_ns);
    if (BuildCombineResponses(batch_tasks, outhost_buffer_, batch_result, input_offset,
                              outhost_line_size_, timestamps) != RET_OK) {
        FreeHostResourcesV2(outhost_buffer_);
        return RET_ERR;
    }

    FreeHostResourcesV2(outhost_buffer_);
    return RET_OK;
}

// 构建响应
int Inference::BuildComblineResponse(TRITONBACKEND_Request *request, const std::vector<void *> &outhost_buffer_,
                                     const std::vector<int64_t> &outsize, int batch_total, int input_offset,
                                     const std::vector<int64_t> &outhost_line_size_,
                                     RequestTimestamps &timestamps)

{
    TRITONBACKEND_Response *response;
    TRITONBACKEND_ResponseNew(&response, request);

    for (size_t j = 0; j < model_state_->GetOutputCount(); j++) {
        void *outhost_buffer = (BYTE_PTR)(outhost_buffer_[j] + input_offset * outhost_line_size_[j]);
        if (BuildSingleCombineOutput(response, j, outhost_buffer, outsize[j], batch_total) != RET_OK) {
            TRITONBACKEND_ResponseDelete(response);
            return RET_ERR;
        }
    }

    // 输出已写入 Triton 响应缓冲区，此时才算本次执行结束（与 onnxruntime 后端一致）
    SET_TIMESTAMP(timestamps.exec_end_ns);

    TRITONSERVER_Error *error = TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr);
    if (error != nullptr) {
        TRITONBACKEND_ResponseDelete(response);
        TRITONSERVER_ErrorDelete(error);
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("TRITONBACKEND_ResponseSend failed, ")).c_str());
        return RET_ERR;
    }
    // 响应发送成功后、释放请求之前上报本请求的耗时统计
    if (model_instance_ != nullptr) {
        LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportStatistics(
            model_instance_, request, true, timestamps.exec_start_ns, timestamps.compute_start_ns,
            timestamps.compute_end_ns, timestamps.exec_end_ns),
            "failed reporting request statistics");
    }
    LOG_IF_ERROR(TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL), "failed releasing request");

    return RET_OK;
}

// 构建单个输出
int Inference::BuildSingleCombineOutput(TRITONBACKEND_Response *response, size_t output_index, void *outhost_buffer,
                                        int64_t buffer_size, int batch_total)
{
    string output_name = model_state_->GetOutputClientTensors()[output_index].name_;
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("current  output_name ") + output_name).c_str());

    TRITONBACKEND_Output *output = nullptr;
    uint32_t size = static_cast<uint32_t>(model_state_->GetOutputClientTensors()[output_index].dims_.size());

    if (model_state_->GetInferMode() == ModelState::InferMode::DYNAMICMODEL || model_state_->GetDynamicBatchSupport()) {
        size += 1;
    }

    std::unique_ptr<int64_t[]> shape_out = CreateOutputShape(output_index, size, batch_total);
    if (!shape_out) {
        return RET_ERR;
    }

    TRITONSERVER_Error *error = TRITONBACKEND_ResponseOutput(
        response, &output, model_state_->GetOutputClientTensors()[output_index].name_.c_str(),
        static_cast<TRITONSERVER_DataType>(model_state_->GetOutputClientTensors()[output_index].dtype_),
        shape_out.get(), size);

    if (error != nullptr) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("TRITONBACKEND_ResponseOutput failed, ")).c_str());
        return RET_ERR;
    }

    void *buffer_out;
    TRITONSERVER_MemoryType memory_type_out = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id_out = 0;
    error = TRITONBACKEND_OutputBuffer(output, &buffer_out, buffer_size, &memory_type_out, &memory_type_id_out);
    if (error != nullptr) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("TRITONBACKEND_OutputBuffer failed, ")).c_str());
        return RET_ERR;
    }

    std::copy((BYTE_PTR)outhost_buffer, (BYTE_PTR)outhost_buffer + buffer_size, (BYTE_PTR)buffer_out);
    return RET_OK;
}

// 创建输出形状数组
std::unique_ptr<int64_t[]> Inference::CreateOutputShape(size_t output_index, uint32_t &size, int batch_total)
{
    std::unique_ptr<int64_t[]> shape_out = std::make_unique<int64_t[]>(size);
    if (model_state_->GetInferMode() == ModelState::InferMode::DYNAMICMODEL || model_state_->GetDynamicBatchSupport()) {
        shape_out[0] = batch_total;
        for (size_t k = 0; k < output_tensor_info_[output_index].tensor_dims_.size(); k++) {
            shape_out[k + 1] = output_tensor_info_[output_index].tensor_dims_[k];
        }
    } else {
        for (size_t k = 0; k < size; k++) {
            shape_out[k] = output_tensor_info_[output_index].tensor_dims_[k];
        }
    }

    return shape_out;
}

bool Inference::IsOperator(char c)
{
    return c == '+' || c == '-' || c == '*' || c == '/';
}

// 辅助函数：获取运算符优先级
int Inference::GetPriority(char op)
{
    if (op == '+' || op == '-')
        return PRIORITY_MID;
    if (op == '*' || op == '/')
        return PRIORITY_HIGH;
    return PRIORITY_LOW;
}

// 辅助函数：执行运算
int Inference::Calculate(int a, int b, char op)
{
    switch (op) {
        case '+':
            return a + b;
        case '-':
            return a - b;
        case '*':
            return a * b;
        case '/':
            if (b == 0)
                throw runtime_error("Division by zero");
            return a / b;
        default:
            throw runtime_error("Invalid operator");
    }
}

// 主函数：计算表达式
int Inference::GetDimNum(const std::map<std::string, int> &values, const std::string &formula)
{
    // 步骤1：替换变量名为对应的数值
    string processedFormula = ReplaceVariables(formula, values);

    // 步骤2：将中缀表达式转换为后缀表达式
    vector<string> postfix = InfixToPostfix(processedFormula);

    // 步骤3：计算后缀表达式
    return EvaluatePostfix(postfix);
}

std::string Inference::ReplaceVariables(const std::string &formula, const std::map<std::string, int> &values)
{
    string result = formula;
    auto isIdentifierChar = [](unsigned char c) { return std::isalnum(c) || c == '_'; };
    for (auto &pair : values) {
        string variable = pair.first;
        string value = to_string(pair.second);

        size_t pos = 0;
        while ((pos = result.find(variable, pos)) != string::npos) {
            bool validBefore = (pos == 0 || !isIdentifierChar(static_cast<unsigned char>(result[pos - 1])));
            bool validAfter = (pos + variable.length() == result.length() ||
                               !isIdentifierChar(static_cast<unsigned char>(result[pos + variable.length()])));

            if (validBefore && validAfter) {
                result.replace(pos, variable.length(), value);
                pos += value.length();
            } else {
                pos += variable.length();
            }
        }
    }
    return result;
}

std::vector<std::string> Inference::InfixToPostfix(const std::string &formula)
{
    stack<char> opStack;
    vector<string> output;
    string currentNumber;

    ProcessFormulaCharacters(formula, opStack, output, currentNumber);
    HandleRemainingOperators(opStack, output);

    return output;
}

void Inference::ProcessFormulaCharacters(const std::string &formula, std::stack<char> &opStack,
                                         std::vector<std::string> &output, std::string &currentNumber)
{
    for (size_t i = 0; i < formula.length(); i++) {
        char c = formula[i];

        if (isdigit(c)) {
            // 处理数字（可能有多位）
            currentNumber += c;
        } else {
            ProcessCurrentNumber(currentNumber, output);
            ProcessCharacter(c, opStack, output);
        }
    }

    // 处理最后一个数字
    if (!currentNumber.empty()) {
        output.push_back(currentNumber);
    }
}

void Inference::ProcessCurrentNumber(std::string &currentNumber, std::vector<std::string> &output)
{
    if (!currentNumber.empty()) {
        output.push_back(currentNumber);
        currentNumber.clear();
    }
}

void Inference::ProcessCharacter(char c, std::stack<char> &opStack, std::vector<std::string> &output)
{
    if (c == '(') {
        opStack.push(c);
    } else if (c == ')') {
        ProcessRightParenthesis(opStack, output);
    } else if (IsOperator(c)) {
        ProcessOperator(c, opStack, output);
    }
    // 忽略空格等其他字符
}

void Inference::ProcessRightParenthesis(std::stack<char> &opStack, std::vector<std::string> &output)
{
    // 弹出直到遇到左括号
    while (!opStack.empty() && opStack.top() != '(') {
        output.push_back(string(1, opStack.top()));
        opStack.pop();
    }
    if (!opStack.empty() && opStack.top() == '(') {
        opStack.pop();  // 弹出左括号
    }
}

void Inference::ProcessOperator(char c, std::stack<char> &opStack, std::vector<std::string> &output)
{
    // 处理运算符优先级
    while (!opStack.empty() && GetPriority(opStack.top()) >= GetPriority(c)) {
        output.push_back(string(1, opStack.top()));
        opStack.pop();
    }
    opStack.push(c);
}

void Inference::HandleRemainingOperators(std::stack<char> &opStack, std::vector<std::string> &output)
{
    // 弹出栈中剩余的所有运算符
    while (!opStack.empty()) {
        output.push_back(string(1, opStack.top()));
        opStack.pop();
    }
}

int Inference::EvaluatePostfix(const std::vector<std::string> &postfix)
{
    stack<int> calcStack;

    for (const string &token : postfix) {
        if (isdigit(token[0]) || (token.length() > 1 && isdigit(token[1]))) {
            // 如果是数字，压入栈中
            calcStack.push(stoi(token));
        } else if (IsOperator(token[0])) {
            // 如果是运算符，弹出两个操作数进行计算
            if (calcStack.size() < OPERATOR_PAIR) {
                throw runtime_error("Invalid expression");
            }
            int b = calcStack.top();
            calcStack.pop();
            int a = calcStack.top();
            calcStack.pop();
            int result = Calculate(a, b, token[0]);
            calcStack.push(result);
        }
    }

    if (calcStack.size() != 1) {
        throw runtime_error("Invalid expression");
    }

    return calcStack.top();
}

void Inference::PrintOutputInfo(
    const std::map<std::pair<size_t, size_t>, triton::backend::npu_ge::ModelState::Express> &m1)
{
    for (auto &p1 : m1) {
        const pair<size_t, size_t> &index = p1.first;
        const ModelState::Express &ex = p1.second;
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "----------------------------------------");
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("Output Tensor") + std::to_string(index.first) +
                                               std::string(" dim") + std::to_string(index.second) + " need calc")
                                                  .c_str());
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("ex.expressName ") + ex.expressName).c_str());
        for (auto &m : ex.dimMap) {
            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("Tensor name: ") + m.first).c_str());
            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("input tensor") + std::to_string(m.second.first) +
                                                   std::string(" dim") + std::to_string(m.second.second))
                                                      .c_str());
        }
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "----------------------------------------");
    }
}

void Inference::CalcNegativeOne()
{
    // 打印输出张量初始维度
    PrintOutputDimensions();

    std::map<std::pair<size_t, size_t>, triton::backend::npu_ge::ModelState::Express> m1 =
        model_state_->GetInToOutMap();
    if (m1.size() == 0) {
        return;
    }

    PrintOutputInfo(m1);

    ProcessMapEntries(m1);

    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "Calculation of -1 completed");

    // 打印输出张量最终维度
    PrintOutputDimensions();
}

void Inference::PrintOutputDimensions()
{
    for (size_t i = 0; i < model_state_->GetOutputCount(); i++) {
        if (output_tensor_info_[i].tensor_dims_.size() == 0) {
            output_tensor_info_[i].tensor_dims_ = model_state_->GetOutputClientTensors()[i].dims_;
        }
        std::string output_dims_str = "Output tensor " + std::to_string(i) + " dimensions: [";
        for (size_t j = 0; j < output_tensor_info_[i].tensor_dims_.size(); j++) {
            output_dims_str += std::to_string(output_tensor_info_[i].tensor_dims_[j]) + " ";
        }
        output_dims_str += "]";
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, output_dims_str.c_str());
    }
}

void Inference::ProcessMapEntries(
    const std::map<std::pair<size_t, size_t>, triton::backend::npu_ge::ModelState::Express> &m1)
{
    for (auto &p1 : m1) {
        pair<size_t, size_t> index = p1.first;
        const ModelState::Express &ex = p1.second;
        map<string, int> values1;

        if (model_state_->GetModelNode() ==
            ModelState::ModelMode::NO_MAX_BATCH_FIRST_SAME_NEGATIVE_ONE_HAVE_UNKNOW_DIM ||
            model_state_->GetModelNode() == ModelState::ModelMode::MAX_BATCH_HAVE_UNKNOW_DIM) {
            ProcessValuesWithBatchOffset(ex, values1);
        } else {
            ProcessValuesWithoutBatchOffset(ex, values1);
        }

        // 计算并设置结果
        output_tensor_info_[index.first].tensor_dims_[index.second] = GetDimNum(values1, ex.expressName);
        LogResult(index);
    }
}

void Inference::ProcessValuesWithBatchOffset(const ModelState::Express &ex, std::map<std::string, int> &values1)
{
    for (auto &pair : ex.dimMap) {
        values1[pair.first] = input_tensor_info_[pair.second.first].tensor_dims_[pair.second.second + 1];
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                    (std::string("tensor name ") + pair.first + std::string(" value: ") +
                     std::to_string(input_tensor_info_[pair.second.first].tensor_dims_[pair.second.second + 1]))
                        .c_str());
    }
}

void Inference::ProcessValuesWithoutBatchOffset(const ModelState::Express &ex, std::map<std::string, int> &values1)
{
    for (auto &pair : ex.dimMap) {
        values1[pair.first] = input_tensor_info_[pair.second.first].tensor_dims_[pair.second.second];
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                    (std::string("tensor name ") + pair.first + std::string(" value: ") +
                     std::to_string(input_tensor_info_[pair.second.first].tensor_dims_[pair.second.second]))
                        .c_str());
    }
}

void Inference::LogResult(const std::pair<size_t, size_t> &index)
{
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("Result output tensor") + std::to_string(index.first) +
                                           std::string(" dim") + std::to_string(index.second) + std::string(" : ") +
                                           std::to_string(output_tensor_info_[index.first].tensor_dims_[index.second]))
                                              .c_str());
}

int Inference::BatchSplicTasks(std::vector<TRITONBACKEND_Request *> &batch_tasks, std::vector<int> &batch_result)
{
    Scheduler *scheduler = model_state_->GetScheduler();
    std::vector<Scheduler::Instance *> instances =
        scheduler->GetIdleInstances(nullptr, model_state_->GetInstanceExecBlock());
    ScopedInstanceStatusReset reset_instances(*scheduler, instances);
    if (ProcessCombineRequestV2(batch_tasks, instances, batch_result) != RET_OK) {
        return RET_ERR;
    }
    return RET_OK;
}

void Inference::SetBatchTaskAndResult(std::vector<TRITONBACKEND_Request *> &batch_tasks, std::vector<int> &batch_result,
                                      TRITONBACKEND_Request **requests, int i)
{
    ModelState::ModelMode modelmode = model_state_->GetModelNode();
    if (modelmode == ModelState::ModelMode::NO_MAX_BATCH_FIRST_NOT_SAME ||
        modelmode == ModelState::ModelMode::NO_MAX_BATCH_FIRST_NOT_SAME_EXIST_NEGATIVE) {
        batch_tasks.push_back(requests[i]);
        batch_result.push_back(1);
        return;
    }
    batch_tasks.push_back(requests[i]);
    TRITONBACKEND_Input *input;
    TRITONBACKEND_RequestInputByIndex(requests[i], 0, &input);

    TRITONSERVER_DataType datatype;
    const int64_t *shape;
    uint32_t dims_count;
    const char *input_name;
    TRITONBACKEND_InputProperties(input, &input_name, &datatype, &shape, &dims_count, nullptr, nullptr);
    batch_result.push_back(shape[0]);
}

int Inference::HandleRequest(TRITONBACKEND_Request **requests, const uint32_t request_count)
{
    if (model_state_->GetDynamicmode().preferred_batch_sizes.size() > 0 &&
        model_state_->GetModelNode() == ModelState::ModelMode::MAX_BATCH && model_state_->GetInToOutMap().size() == 0) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "Batch Request Start");
        std::vector<TRITONBACKEND_Request *> batch_tasks;
        std::vector<int> batch_result;
        for (size_t i = 0; i < request_count; i++) {
            SetBatchTaskAndResult(batch_tasks, batch_result, requests, i);
        }
        if (BatchSplicTasks(batch_tasks, batch_result) != RET_OK) {
            return RET_ERR;
        }
    } else {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "single Request Start");

        for (size_t i = 0; i < request_count; i++) {
            std::vector<TRITONBACKEND_Request *> batch_tasks;
            std::vector<int> batch_result;
            SetBatchTaskAndResult(batch_tasks, batch_result, requests, i);
            if (BatchSplicTasks(batch_tasks, batch_result) != RET_OK) {
                return RET_ERR;
            }
        }
    }
    return RET_OK;
}

}
}
}
