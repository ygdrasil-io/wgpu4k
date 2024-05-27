package io.ygdrasil.wgpu

actual class Buffer : AutoCloseable {

    actual val size: GPUSize64
        get() = TODO("not yet implemented")


    actual fun unmap() {
        TODO("not yet implemented")
    }

    actual fun map(buffer: FloatArray) {
        TODO("not yet implemented")
    }

    actual fun getMappedRange(offset: GPUSize64?, size: GPUSize64?): ByteArray {
        TODO("not yet implemented")
    }

    actual override fun close() {
        TODO("Not yet implemented")
    }

}