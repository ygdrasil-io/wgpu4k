package io.ygdrasil.webgpu

import ffi.ArrayHolder
import ffi.NativeAddress
import ffi.memoryScope
import io.ygdrasil.webgpu.mapper.map
import io.ygdrasil.wgpu.WGPUCommandBuffer
import io.ygdrasil.wgpu.WGPUQueue
import io.ygdrasil.wgpu.WGPUStringView
import io.ygdrasil.wgpu.wgpuQueueSetLabel
import io.ygdrasil.wgpu.wgpuQueueSubmit
import io.ygdrasil.wgpu.wgpuQueueWriteBuffer
import io.ygdrasil.wgpu.wgpuQueueWriteTexture

actual class Queue(val handler: WGPUQueue) : GPUQueue {

    actual override var label: String
        get() = TODO("Not yet implemented")
        set(value) = memoryScope { scope ->
            val newLabel = WGPUStringView.allocate(scope)
                .also { scope.map(value, it) }
            wgpuQueueSetLabel(handler, newLabel)
        }

    actual override fun submit(commandBuffers: List<GPUCommandBuffer>)= memoryScope { scope ->
        if (commandBuffers.isNotEmpty()) {

            val commands = scope.bufferOfAddresses(commandBuffers.map { (it as CommandBuffer).handler.handler })
                .handler
                .let { ArrayHolder<WGPUCommandBuffer>(it) }

            wgpuQueueSubmit(
                handler,
                commandBuffers.size.toULong(),
                commands
            )
        } else {

            wgpuQueueSubmit(
                handler,
                0uL,
                null
            )
        }
    }

    actual override suspend fun onSubmittedWorkDone(): Result<Unit> {
        TODO("Not yet implemented")
    }

    actual override fun writeBuffer(
        buffer: GPUBuffer,
        bufferOffset: GPUSize64,
        data: ArrayBuffer,
        dataOffset: GPUSize64,
        size: GPUSize64?
    ) {
        val size = size ?: (buffer.size - bufferOffset)
        val data = (data.rawPointer + dataOffset).toNativeAddress()
        wgpuQueueWriteBuffer(handler, (buffer as Buffer).handler, bufferOffset, data, size)
    }

    actual override fun writeTexture(
        destination: GPUTexelCopyTextureInfo,
        data: ArrayBuffer,
        dataLayout: GPUTexelCopyBufferLayout,
        size: GPUExtent3D
    ) = memoryScope { scope ->
        val dataSize = TODO("missing size on ArrayBuffer")
        wgpuQueueWriteTexture(handler, scope.map(destination), data.rawPointer.toNativeAddress(), 0uL, scope.map(dataLayout), scope.map(size))
    }

}

internal expect fun ULong.toNativeAddress(): NativeAddress?
