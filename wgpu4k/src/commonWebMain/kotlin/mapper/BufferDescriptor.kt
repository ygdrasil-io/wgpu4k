package io.ygdrasil.webgpu.mapper

import io.ygdrasil.webgpu.GPUBufferDescriptor
import io.ygdrasil.webgpu.WGPUBufferDescriptor
import io.ygdrasil.webgpu.asJsNumber
import io.ygdrasil.webgpu.createJsObject

internal fun map(input: GPUBufferDescriptor): WGPUBufferDescriptor = createJsObject<WGPUBufferDescriptor>().apply {
    label = input.label
    size = input.size.asJsNumber()
    usage = input.usage.value.asJsNumber()
    mappedAtCreation = input.mappedAtCreation
}
