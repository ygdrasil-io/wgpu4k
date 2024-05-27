package io.ygdrasil.wgpu

/**
 * A GPUSampler encodes transformations and filtering information that can be used in a shader to interpret texture resource data.
 *
 * @see <a href="https://www.w3.org/TR/webgpu/#gpusampler">W3C specifications</a>
 */
actual class Sampler : AutoCloseable {
    actual override fun close() {
        TODO("Not yet implemented")
    }
}