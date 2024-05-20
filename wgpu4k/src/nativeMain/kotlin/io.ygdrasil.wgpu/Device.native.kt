@file:OptIn(ExperimentalForeignApi::class)

package io.ygdrasil.wgpu

import kotlinx.cinterop.ExperimentalForeignApi
import webgpu.WGPUDevice
import webgpu.wgpuDeviceCreateShaderModule

actual class Device(val handler: WGPUDevice) : AutoCloseable {
    actual val queue: Queue
        get() = TODO("Not yet implemented")

    actual fun createCommandEncoder(descriptor: CommandEncoderDescriptor?): CommandEncoder {
        TODO("Not yet implemented")
    }

    actual fun createShaderModule(descriptor: ShaderModuleDescriptor): ShaderModule {
        wgpuDeviceCreateShaderModule()
        TODO("Not yet implemented")
    }

    actual fun createPipelineLayout(descriptor: PipelineLayoutDescriptor): PipelineLayout {
        TODO("Not yet implemented")
    }

    actual fun createRenderPipeline(descriptor: RenderPipelineDescriptor): RenderPipeline {
        TODO("Not yet implemented")
    }

    actual fun createBuffer(descriptor: BufferDescriptor): Buffer {
        TODO("Not yet implemented")
    }

    actual fun createTexture(descriptor: TextureDescriptor): Texture {
        TODO("Not yet implemented")
    }

    actual fun createBindGroup(descriptor: BindGroupDescriptor): BindGroup {
        TODO("Not yet implemented")
    }

    actual fun createSampler(descriptor: SamplerDescriptor): Sampler {
        TODO("Not yet implemented")
    }

    actual fun createComputePipeline(descriptor: ComputePipelineDescriptor): ComputePipeline {
        TODO("Not yet implemented")
    }

    override fun close() {
        TODO("Not yet implemented")
    }

}