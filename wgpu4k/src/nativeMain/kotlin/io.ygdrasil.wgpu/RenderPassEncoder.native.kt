package io.ygdrasil.wgpu

actual class RenderPassEncoder: AutoCloseable {

    actual fun end() {
        TODO("Not yet implemented")
    }

    actual fun setPipeline(renderPipeline: RenderPipeline) {
        TODO("Not yet implemented")
    }

    actual fun draw(
        vertexCount: GPUSize32,
        instanceCount: GPUSize32,
        firstVertex: GPUSize32,
        firstInstance: GPUSize32
    ) {
        TODO("Not yet implemented")
    }

    actual fun setBindGroup(index: Int, bindGroup: BindGroup) {
        TODO("Not yet implemented")
    }

    actual fun setVertexBuffer(slot: Int, buffer: Buffer) {
        TODO("Not yet implemented")
    }

    override fun close() {
        TODO("Not yet implemented")
    }

}