@file:OptIn(ExperimentalForeignApi::class)

package io.ygdrasil.wgpu

import kotlinx.cinterop.ExperimentalForeignApi
import webgpu.WGPUTexture

actual class Texture(val handler: WGPUTexture): AutoCloseable {

    actual val width: GPUIntegerCoordinateOut
        get() = TODO("not yet implemented")
    actual val height: GPUIntegerCoordinateOut
        get() = TODO("not yet implemented")
    actual val depthOrArrayLayers: GPUIntegerCoordinateOut
        get() = TODO("not yet implemented")
    actual val mipLevelCount: GPUIntegerCoordinateOut
        get() = TODO("not yet implemented")
    actual val sampleCount: GPUSize32Out
        get() = TODO("not yet implemented")
    actual val dimension: TextureDimension
        get() = TODO("not yet implemented")
    actual val format: TextureFormat
        get() = TODO("not yet implemented")
    actual val usage: GPUFlagsConstant
        get() = TODO("not yet implemented")

    actual fun createView(descriptor: TextureViewDescriptor?): TextureView {
        TODO("not yet implemented")
    }

    actual override fun close() {
        TODO("Not yet implemented")
    }
}