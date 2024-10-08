package io.ygdrasil.wgpu.mapper

import io.ygdrasil.wgpu.BindGroupLayoutDescriptor
import io.ygdrasil.wgpu.internal.jna.WGPUBindGroupLayoutDescriptor
import io.ygdrasil.wgpu.internal.jna.WGPUBindGroupLayoutEntry
import io.ygdrasil.wgpu.internal.jna.WGPUBufferBindingLayout
import io.ygdrasil.wgpu.internal.jna.WGPUChainedStruct
import io.ygdrasil.wgpu.internal.jna.WGPUSamplerBindingLayout
import io.ygdrasil.wgpu.internal.jna.WGPUStorageTextureBindingLayout
import io.ygdrasil.wgpu.internal.jna.WGPUTextureBindingLayout
import io.ygdrasil.wgpu.internal.jna.wgpu_h
import io.ygdrasil.wgpu.internal.toAddress
import io.ygdrasil.wgpu.toFlagInt
import io.ygdrasil.wgpu.toInt
import java.lang.foreign.MemorySegment
import java.lang.foreign.SegmentAllocator

fun SegmentAllocator.map(input: BindGroupLayoutDescriptor) =
    WGPUBindGroupLayoutDescriptor.allocate(this).also { output ->
    if (input.label != null) WGPUBindGroupLayoutDescriptor.label(output, allocateFrom(input.label))

    if (input.entries.isNotEmpty()) {
        WGPUBindGroupLayoutDescriptor.entryCount(output, input.entries.size.toLong())
        val entries = WGPUBindGroupLayoutEntry.allocateArray(input.entries.size.toLong(), this)
        input.entries.forEachIndexed { index, entry ->
            map(entry, WGPUBindGroupLayoutEntry.asSlice(entries, index.toLong()))
        }
        WGPUBindGroupLayoutDescriptor.entries(output, entries)
    }
    }.pointer.toAddress()

fun SegmentAllocator.map(input: BindGroupLayoutDescriptor.Entry, output: MemorySegment) {

    WGPUBindGroupLayoutEntry.binding(output, input.binding)
    WGPUBindGroupLayoutEntry.visibility(output, input.visibility.toFlagInt())

    when (val bindingType = input.bindingType) {
        is BindGroupLayoutDescriptor.Entry.BufferBindingLayout -> {
            val buffer = WGPUBindGroupLayoutEntry.buffer(output)
            WGPUBufferBindingLayout.hasDynamicOffset(buffer, bindingType.hasDynamicOffset.toInt())
            WGPUBufferBindingLayout.minBindingSize(buffer, bindingType.minBindingSize)
            WGPUBufferBindingLayout.type(buffer, bindingType.type.value)

            val chain = WGPUChainedStruct.allocate(this)
            WGPUChainedStruct.sType(chain, wgpu_h.WGPUSType_BindGroupEntryExtras())
            WGPUBufferBindingLayout.nextInChain(buffer, chain)
        }

        is BindGroupLayoutDescriptor.Entry.SamplerBindingLayout -> {
            val sampler = WGPUBindGroupLayoutEntry.sampler(output)
            WGPUSamplerBindingLayout.type(sampler, bindingType.type.value)

            val chain = WGPUChainedStruct.allocate(this)
            WGPUChainedStruct.sType(chain, wgpu_h.WGPUSType_BindGroupEntryExtras())
            WGPUSamplerBindingLayout.nextInChain(sampler, chain)
        }

        is BindGroupLayoutDescriptor.Entry.TextureBindingLayout -> {
            val texture = WGPUBindGroupLayoutEntry.texture(output)
            WGPUTextureBindingLayout.multisampled(texture, bindingType.multisampled.toInt())
            WGPUTextureBindingLayout.sampleType(texture, bindingType.sampleType.value)
            WGPUTextureBindingLayout.viewDimension(texture, bindingType.viewDimension.value)

            val chain = WGPUChainedStruct.allocate(this)
            WGPUChainedStruct.sType(chain, wgpu_h.WGPUSType_BindGroupEntryExtras())
            WGPUTextureBindingLayout.nextInChain(texture, chain)
        }

        is BindGroupLayoutDescriptor.Entry.StorageTextureBindingLayout -> {
            val storageTexture = WGPUBindGroupLayoutEntry.storageTexture(output)
            WGPUStorageTextureBindingLayout.access(storageTexture, bindingType.access.value)
            WGPUStorageTextureBindingLayout.format(storageTexture, bindingType.format.value)
            WGPUStorageTextureBindingLayout.viewDimension(storageTexture, bindingType.viewDimension.value)

            val chain = WGPUChainedStruct.allocate(this)
            WGPUChainedStruct.sType(chain, wgpu_h.WGPUSType_BindGroupEntryExtras())
            WGPUStorageTextureBindingLayout.nextInChain(storageTexture, chain)

        }
    }

}