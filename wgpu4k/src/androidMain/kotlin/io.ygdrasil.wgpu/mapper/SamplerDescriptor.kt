package io.ygdrasil.wgpu.mapper

import io.ygdrasil.wgpu.SamplerDescriptor
import io.ygdrasil.wgpu.internal.jna.WGPUSamplerDescriptor
import io.ygdrasil.wgpu.internal.toAddress
import java.lang.foreign.MemorySegment
import java.lang.foreign.SegmentAllocator

internal fun SegmentAllocator.map(input: SamplerDescriptor): Long =
    WGPUSamplerDescriptor.allocate(this).also { output ->
    if (input.label != null) WGPUSamplerDescriptor.label(output, allocateFrom(input.label))

    WGPUSamplerDescriptor.addressModeU(output, input.addressModeU.value)
    WGPUSamplerDescriptor.addressModeV(output, input.addressModeV.value)
    WGPUSamplerDescriptor.addressModeW(output, input.addressModeW.value)

    WGPUSamplerDescriptor.magFilter(output, input.magFilter.value)
    WGPUSamplerDescriptor.minFilter(output, input.minFilter.value)
    WGPUSamplerDescriptor.mipmapFilter(output, input.mipmapFilter.value)

    WGPUSamplerDescriptor.lodMinClamp(output, input.lodMinClamp)
    WGPUSamplerDescriptor.lodMaxClamp(output, input.lodMaxClamp)

    if (input.compare != null) WGPUSamplerDescriptor.compare(output, input.compare.value)
    WGPUSamplerDescriptor.maxAnisotropy(output, input.maxAnisotropy.toShort())
    }.pointer.toAddress()
