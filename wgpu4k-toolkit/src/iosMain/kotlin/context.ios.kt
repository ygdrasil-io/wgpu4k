@file:OptIn(ExperimentalForeignApi::class)

package io.ygdrasil.webgpu

import ffi.NativeAddress
import kotlinx.cinterop.COpaque
import kotlinx.cinterop.COpaquePointer
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.interpretCPointer
import kotlinx.cinterop.objcPtr
import kotlinx.cinterop.reinterpret
import platform.MetalKit.MTKView


suspend fun iosContextRenderer(view: MTKView, width: Int, height: Int, deferredRendering: Boolean = false): IosContext {
    val layer = view.layer
    val layerPointer: COpaquePointer = interpretCPointer<COpaque>(layer.objcPtr())!!.reinterpret()
    val instance = WGPU.createInstance() ?: error("Can't create WGPU instance")
    val nativeSurface = instance.getSurfaceFromMetalLayer(layerPointer.let(::NativeAddress)) ?: error("Can't create Surface")
    val adapter = instance.requestAdapter(nativeSurface) ?: error("Can't create Adapter")
    val device = adapter.requestDevice().getOrThrow()
    val surface = Surface(nativeSurface, width.toUInt(), height.toUInt())

    nativeSurface.computeSurfaceCapabilities(adapter)

    val renderingContext = when (deferredRendering) {
        true -> TextureRenderingContext(width.toUInt(), height.toUInt(), GPUTextureFormat.RGBA8Unorm, device)
        false -> SurfaceRenderingContext(surface, surface.supportedFormats.first())
    }

    return IosContext(
        view,
        WGPUContext(surface, adapter, device, renderingContext)
    )
}


class IosContext(
    val view: MTKView,
    val wgpuContext: WGPUContext,
) : AutoCloseable {

    override fun close() {
        wgpuContext.close()
    }
}