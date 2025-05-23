package io.ygdrasil.webgpu

import ffi.memoryScope
import io.ygdrasil.webgpu.mapper.errorOf
import io.ygdrasil.webgpu.mapper.map
import io.ygdrasil.wgpu.WGPUCommandEncoderDescriptor
import io.ygdrasil.wgpu.WGPUCreateComputePipelineAsyncCallback
import io.ygdrasil.wgpu.WGPUCreateComputePipelineAsyncCallbackInfo
import io.ygdrasil.wgpu.WGPUCreatePipelineAsyncStatus_Success
import io.ygdrasil.wgpu.WGPUCreateRenderPipelineAsyncCallback
import io.ygdrasil.wgpu.WGPUCreateRenderPipelineAsyncCallbackInfo
import io.ygdrasil.wgpu.WGPUDevice
import io.ygdrasil.wgpu.WGPULimits
import io.ygdrasil.wgpu.WGPUPopErrorScopeCallback
import io.ygdrasil.wgpu.WGPUPopErrorScopeCallbackInfo
import io.ygdrasil.wgpu.WGPUPopErrorScopeStatus_Success
import io.ygdrasil.wgpu.WGPUStringView
import io.ygdrasil.wgpu.wgpuDeviceCreateBindGroup
import io.ygdrasil.wgpu.wgpuDeviceCreateBindGroupLayout
import io.ygdrasil.wgpu.wgpuDeviceCreateBuffer
import io.ygdrasil.wgpu.wgpuDeviceCreateCommandEncoder
import io.ygdrasil.wgpu.wgpuDeviceCreateComputePipeline
import io.ygdrasil.wgpu.wgpuDeviceCreateComputePipelineAsync
import io.ygdrasil.wgpu.wgpuDeviceCreatePipelineLayout
import io.ygdrasil.wgpu.wgpuDeviceCreateQuerySet
import io.ygdrasil.wgpu.wgpuDeviceCreateRenderBundleEncoder
import io.ygdrasil.wgpu.wgpuDeviceCreateRenderPipeline
import io.ygdrasil.wgpu.wgpuDeviceCreateRenderPipelineAsync
import io.ygdrasil.wgpu.wgpuDeviceCreateSampler
import io.ygdrasil.wgpu.wgpuDeviceCreateShaderModule
import io.ygdrasil.wgpu.wgpuDeviceCreateTexture
import io.ygdrasil.wgpu.wgpuDeviceGetAdapterInfo
import io.ygdrasil.wgpu.wgpuDeviceGetLimits
import io.ygdrasil.wgpu.wgpuDeviceGetQueue
import io.ygdrasil.wgpu.wgpuDeviceHasFeature
import io.ygdrasil.wgpu.wgpuDevicePopErrorScope
import io.ygdrasil.wgpu.wgpuDevicePushErrorScope
import io.ygdrasil.wgpu.wgpuDeviceRelease
import io.ygdrasil.wgpu.wgpuDeviceSetLabel
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine

actual class Device(val handler: WGPUDevice, label: String) : GPUDevice {

    actual override var label: String = label
        set(value) = memoryScope { scope ->
            val newLabel = WGPUStringView.allocate(scope)
                .also { scope.map(value, it) }
            wgpuDeviceSetLabel(handler, newLabel)
            field = value
        }

    actual override val queue: GPUQueue by lazy { Queue(wgpuDeviceGetQueue(handler) ?: error("fail to get device queue"), "") }

    actual override val features: Set<GPUFeatureName> by lazy {
        GPUFeatureName.entries
            .mapNotNull { feature ->
                feature.takeIf { wgpuDeviceHasFeature(handler, feature.value) }
            }
            .toSet()
    }

    actual override val limits: GPUSupportedLimits
        get() = memoryScope { scope ->
            val supportedLimits = WGPULimits.allocate(scope)
            wgpuDeviceGetLimits(handler, supportedLimits)
            map(supportedLimits)
        }

    actual override val adapterInfo: GPUAdapterInfo
        get() = map(wgpuDeviceGetAdapterInfo(handler))

    actual override fun createCommandEncoder(descriptor: GPUCommandEncoderDescriptor?): GPUCommandEncoder = memoryScope { scope ->
        WGPUCommandEncoderDescriptor.allocate(scope)
            .let { wgpuDeviceCreateCommandEncoder(handler, it) }
            ?.let {CommandEncoder(it, descriptor?.label ?: "")} ?: error("fail to create command encoder")
    }

    actual override fun createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule = memoryScope { scope ->
        scope.map(descriptor)
            .let { wgpuDeviceCreateShaderModule(handler, it) }
            ?.let { ShaderModule(it, descriptor.label)} ?: error("fail to create shader module")
    }

    actual override fun createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor): GPUPipelineLayout = memoryScope { scope ->
        scope.map(descriptor)
            .let { wgpuDeviceCreatePipelineLayout(handler, it) }
            ?.let { PipelineLayout(it, descriptor.label)} ?: error("fail to create pipeline layout")
    }

    actual override fun createRenderPipeline(descriptor: GPURenderPipelineDescriptor): GPURenderPipeline = memoryScope { scope ->
        scope.map(descriptor)
            .let { wgpuDeviceCreateRenderPipeline(handler, it) }
            ?.let { RenderPipeline(it, descriptor.label)} ?: error("fail to create render pipeline")
    }

    actual override suspend fun createComputePipelineAsync(descriptor: GPUComputePipelineDescriptor): Result<GPUComputePipeline> = suspendCoroutine { continuation ->
        memoryScope { scope ->

            val callback = WGPUCreateComputePipelineAsyncCallback.allocate(scope) { status, pipeline, message, userdata1, userdata2 ->
                continuation.resume(when(status) {
                    WGPUCreatePipelineAsyncStatus_Success -> when (pipeline) {
                        null -> Result.failure(IllegalStateException("ComputePipeline is null"))
                        else -> Result.success(ComputePipeline(pipeline, descriptor.label))
                    }
                    else -> Result.failure(IllegalStateException("request ComputePipeline fail with status: $status and message: ${message?.data?.toKString(message.length)}"))
                })
            }

            val callbackInfo = WGPUCreateComputePipelineAsyncCallbackInfo.allocate(scope).apply {
                this.callback = callback
                this.userdata2 = scope.bufferOfAddress(callback.handler).handler
            }

            wgpuDeviceCreateComputePipelineAsync(handler, scope.map(descriptor), callbackInfo)
        }
    }

    actual override suspend fun createRenderPipelineAsync(descriptor: GPURenderPipelineDescriptor): Result<GPURenderPipeline> = suspendCoroutine { continuation ->
        memoryScope { scope ->

            val callback =
                WGPUCreateRenderPipelineAsyncCallback.allocate(scope) { status, pipeline, message, userdata1, userdata2 ->
                continuation.resume(when(status) {
                    WGPUCreatePipelineAsyncStatus_Success -> when (pipeline) {
                        null -> Result.failure(IllegalStateException("RenderPipeline is null"))
                        else -> Result.success(RenderPipeline(pipeline, descriptor.label))
                    }
                    else -> Result.failure(IllegalStateException("request RenderPipeline fail with status: $status and message: ${message?.data?.toKString(message.length)}"))
                })
            }

            val callbackInfo = WGPUCreateRenderPipelineAsyncCallbackInfo.allocate(scope).apply {
                this.callback = callback
                this.userdata2 = scope.bufferOfAddress(callback.handler).handler
            }

            wgpuDeviceCreateRenderPipelineAsync(handler, scope.map(descriptor), callbackInfo)
        }
    }

    actual override fun createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer = memoryScope { scope ->
        scope.map(descriptor)
            .let { wgpuDeviceCreateBuffer(handler, it) }
            ?.let { Buffer(it, this, descriptor.label)} ?: error("fail to create buffer")
    }

    actual override fun createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup = memoryScope { scope ->
        scope.map(descriptor)
            .let { wgpuDeviceCreateBindGroup(handler, it) }
            ?.let { BindGroup(it, descriptor.label)} ?: error("fail to create bind group")
    }

    actual override fun createTexture(descriptor: GPUTextureDescriptor): GPUTexture = memoryScope { scope ->
        scope.map(descriptor)
            .let { wgpuDeviceCreateTexture(handler, it) }
            ?.let { Texture(it, descriptor.label) } ?: error("fail to create texture")
    }

    actual override fun createSampler(descriptor: GPUSamplerDescriptor?): GPUSampler = memoryScope { scope ->
        descriptor?.let { scope.map(descriptor) }
            .let { wgpuDeviceCreateSampler(handler, it) }
            ?.let { Sampler(it, descriptor?.label ?: "")} ?: error("fail to create sampler")
    }

    actual override fun createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline = memoryScope { scope ->
        scope.map(descriptor)
            .let { wgpuDeviceCreateComputePipeline(handler, it) }
            ?.let { ComputePipeline(it, descriptor.label) } ?: error("fail to create compute pipeline")
    }

    actual override fun createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout = memoryScope { scope ->
        scope.map(descriptor)
            .let { wgpuDeviceCreateBindGroupLayout(handler, it) }
            ?.let { BindGroupLayout(it, descriptor.label)} ?: error("fail to create bind group layout")
    }

    actual override fun createRenderBundleEncoder(descriptor: GPURenderBundleEncoderDescriptor): GPURenderBundleEncoder =
        memoryScope { scope ->
            scope.map(descriptor)
                .let { wgpuDeviceCreateRenderBundleEncoder(handler, it) }
                ?.let { RenderBundleEncoder(it, descriptor.label)} ?: error("fail to create bind group layout")
        }

    actual override fun createQuerySet(descriptor: GPUQuerySetDescriptor): GPUQuerySet = memoryScope { scope ->
        scope.map(descriptor)
            .let { wgpuDeviceCreateQuerySet(handler, it) }
            ?.let { QuerySet(it, descriptor.label)} ?: error("fail to create bind group layout")
    }

    actual override fun pushErrorScope(filter: GPUErrorFilter) {
        wgpuDevicePushErrorScope(handler, filter.value)
    }

    actual override suspend fun popErrorScope(): Result<GPUError?>  = suspendCoroutine { continuation ->
        memoryScope { scope ->

            val callback = WGPUPopErrorScopeCallback.allocate(scope) { status, error, message, userdata1, userdata2 ->
                val message = message?.data?.toKString(message.length)
                continuation.resume(when(status) {
                        WGPUPopErrorScopeStatus_Success -> when (error) {
                            else -> Result.success(errorOf(error, message))
                        }
                        else -> Result.failure(IllegalStateException("request GPUError fail with status: $status and message: $message"))
                    })
                }

            val callbackInfo = WGPUPopErrorScopeCallbackInfo.allocate(scope).apply {
                this.callback = callback
                this.userdata2 = scope.bufferOfAddress(callback.handler).handler
            }

            wgpuDevicePopErrorScope(handler, callbackInfo)
        }
    }

    actual override fun close() {
        wgpuDeviceRelease(handler)
    }
}