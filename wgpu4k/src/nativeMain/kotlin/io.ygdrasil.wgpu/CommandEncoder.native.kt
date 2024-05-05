package io.ygdrasil.wgpu

actual class CommandEncoder : AutoCloseable {

    actual fun beginRenderPass(descriptor: RenderPassDescriptor): RenderPassEncoder {
        TODO("Not yet implemented")
    }

    actual fun finish(): CommandBuffer {
        TODO("Not yet implemented")
    }

    actual fun copyTextureToTexture(
        source: ImageCopyTexture,
        destination: ImageCopyTexture,
        copySize: GPUIntegerCoordinates
    ) {
        TODO("Not yet implemented")
    }

    override fun close() {
        TODO("Not yet implemented")
    }

    actual fun beginComputePass(descriptor: ComputePassDescriptor?): ComputePassEncoder {
        TODO("Not yet implemented")
    }
}