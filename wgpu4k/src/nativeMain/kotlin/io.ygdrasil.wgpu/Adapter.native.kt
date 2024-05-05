package io.ygdrasil.wgpu

/**
 * The GPUAdapter interface of the WebGPU API represents a GPU adapter.
 * From this you can request a Device, adapter info, features, and limits.
 */
actual class Adapter : AutoCloseable {
    actual suspend fun requestDevice(): Device? {
        TODO("Not yet implemented")
    }

    override fun close() {
        TODO("Not yet implemented")
    }
}