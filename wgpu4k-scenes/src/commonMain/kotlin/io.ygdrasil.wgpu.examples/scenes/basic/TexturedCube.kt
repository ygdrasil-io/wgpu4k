package io.ygdrasil.wgpu.examples.scenes.basic

import io.ygdrasil.wgpu.AutoClosableContext
import io.ygdrasil.wgpu.BindGroup
import io.ygdrasil.wgpu.BindGroupDescriptor
import io.ygdrasil.wgpu.Buffer
import io.ygdrasil.wgpu.BufferDescriptor
import io.ygdrasil.wgpu.BufferUsage
import io.ygdrasil.wgpu.Color
import io.ygdrasil.wgpu.CompareFunction
import io.ygdrasil.wgpu.CullMode
import io.ygdrasil.wgpu.FilterMode
import io.ygdrasil.wgpu.ImageCopyExternalImage
import io.ygdrasil.wgpu.ImageCopyTextureTagged
import io.ygdrasil.wgpu.LoadOp
import io.ygdrasil.wgpu.PrimitiveTopology
import io.ygdrasil.wgpu.RenderPassDescriptor
import io.ygdrasil.wgpu.RenderPipeline
import io.ygdrasil.wgpu.RenderPipelineDescriptor
import io.ygdrasil.wgpu.SamplerDescriptor
import io.ygdrasil.wgpu.ShaderModuleDescriptor
import io.ygdrasil.wgpu.Size3D
import io.ygdrasil.wgpu.StoreOp
import io.ygdrasil.wgpu.TextureDescriptor
import io.ygdrasil.wgpu.TextureFormat
import io.ygdrasil.wgpu.TextureUsage
import io.ygdrasil.wgpu.VertexFormat
import io.ygdrasil.wgpu.WGPUContext
import io.ygdrasil.wgpu.beginRenderPass
import io.ygdrasil.wgpu.examples.AssetManager
import io.ygdrasil.wgpu.examples.Scene
import io.ygdrasil.wgpu.examples.scenes.mesh.Cube
import io.ygdrasil.wgpu.examples.scenes.shader.fragment.sampleTextureMixColorShader
import io.ygdrasil.wgpu.examples.scenes.shader.vertex.basicVertexShader
import korlibs.math.geom.Angle
import korlibs.math.geom.Matrix4
import kotlin.math.PI

class TexturedCubeScene(wgpuContext: WGPUContext, assetManager: AssetManager) : Scene(wgpuContext), AssetManager by assetManager {

    lateinit var renderPipeline: RenderPipeline
    lateinit var projectionMatrix: Matrix4
    lateinit var renderPassDescriptor: RenderPassDescriptor
    lateinit var uniformBuffer: Buffer
    lateinit var uniformBindGroup: BindGroup
    lateinit var verticesBuffer: Buffer

    override suspend fun initialize() = with(autoClosableContext) {

        // Create a vertex buffer from the cube data.
        verticesBuffer = device.createBuffer(
            BufferDescriptor(
                size = (Cube.cubeVertexArray.size * Float.SIZE_BYTES).toLong(),
                usage = setOf(BufferUsage.vertex),
                mappedAtCreation = true
            )
        )

        // Util method to use getMappedRange
        verticesBuffer.mapFrom(Cube.cubeVertexArray)
        verticesBuffer.unmap()

        renderPipeline = device.createRenderPipeline(
            RenderPipelineDescriptor(
                vertex = RenderPipelineDescriptor.VertexState(
                    module = device.createShaderModule(
                        ShaderModuleDescriptor(
                            code = basicVertexShader
                        )
                    ).bind(), // bind to autoClosableContext to release it later
                    buffers = listOf(
                        RenderPipelineDescriptor.VertexState.VertexBufferLayout(
                            arrayStride = Cube.cubeVertexSize,
                            attributes = listOf(
                                RenderPipelineDescriptor.VertexState.VertexBufferLayout.VertexAttribute(
                                    shaderLocation = 0,
                                    offset = Cube.cubePositionOffset,
                                    format = VertexFormat.float32x4
                                ),
                                RenderPipelineDescriptor.VertexState.VertexBufferLayout.VertexAttribute(
                                    shaderLocation = 1,
                                    offset = Cube.cubeUVOffset,
                                    format = VertexFormat.float32x2
                                )
                            )
                        )
                    )
                ),
                fragment = RenderPipelineDescriptor.FragmentState(
                    module = device.createShaderModule(
                        ShaderModuleDescriptor(
                            code = sampleTextureMixColorShader
                        )
                    ).bind(), // bind to autoClosableContext to release it later
                    targets = listOf(
                        RenderPipelineDescriptor.FragmentState.ColorTargetState(
                            format = renderingContext.textureFormat
                        )
                    )
                ),
                primitive = RenderPipelineDescriptor.PrimitiveState(
                    topology = PrimitiveTopology.trianglelist,
                    cullMode = CullMode.back
                ),
                depthStencil = RenderPipelineDescriptor.DepthStencilState(
                    depthWriteEnabled = true,
                    depthCompare = CompareFunction.less,
                    format = TextureFormat.depth24plus
                )
            )
        ).bind()

        val depthTexture = device.createTexture(
            TextureDescriptor(
                size = Size3D(renderingContext.width, renderingContext.height),
                format = TextureFormat.depth24plus,
                usage = setOf(TextureUsage.renderattachment),
            )
        ).bind()

        val uniformBufferSize = 4L * 16L; // 4x4 matrix
        uniformBuffer = device.createBuffer(
            BufferDescriptor(
                size = uniformBufferSize,
                usage = setOf(BufferUsage.uniform, BufferUsage.copydst)
            )
        ).bind()


        // Fetch the image and upload it into a GPUTexture.
        val imageBitmapWidth = 512
        val imageBitmapHeight = 512
        val cubeTexture = device.createTexture(
            TextureDescriptor(
                size = Size3D(imageBitmapWidth, imageBitmapHeight),
                format = renderingContext.textureFormat,
                usage = setOf(TextureUsage.texturebinding, TextureUsage.copydst, TextureUsage.renderattachment),
            )
        )

        device.queue.copyExternalImageToTexture(
            ImageCopyExternalImage(source = Di3d),
            ImageCopyTextureTagged(texture = cubeTexture),
            imageBitmapWidth to imageBitmapHeight
        )

        // Create a sampler with linear filtering for smooth interpolation.
        val sampler = device.createSampler(
            SamplerDescriptor(
                magFilter = FilterMode.linear,
                minFilter = FilterMode.linear,
            )
        )

        uniformBindGroup = device.createBindGroup(
            BindGroupDescriptor(
                layout = renderPipeline.getBindGroupLayout(0),
                entries = listOf(
                    BindGroupDescriptor.BindGroupEntry(
                        binding = 0,
                        resource = BindGroupDescriptor.BufferBinding(
                            buffer = uniformBuffer
                        )
                    ),
                    BindGroupDescriptor.BindGroupEntry(
                        binding = 1,
                        resource = BindGroupDescriptor.SamplerBinding(
                            sampler = sampler
                        )
                    ),
                    BindGroupDescriptor.BindGroupEntry(
                        binding = 2,
                        resource = BindGroupDescriptor.TextureViewBinding(
                            view = cubeTexture.createView()
                        )
                    )
                )
            )
        )

        renderPassDescriptor = RenderPassDescriptor(
            colorAttachments = listOf(
                RenderPassDescriptor.ColorAttachment(
                    view = dummyTexture.createView().bind(), // Assigned later
                    loadOp = LoadOp.clear,
                    clearValue = Color(0.5, 0.5, 0.5, 1.0),
                    storeOp = StoreOp.store,
                )
            ),
            depthStencilAttachment = RenderPassDescriptor.DepthStencilAttachment(
                view = depthTexture.createView(),
                depthClearValue = 1.0f,
                depthLoadOp = LoadOp.clear,
                depthStoreOp = StoreOp.store
            )
        )


        val aspect = renderingContext.width / renderingContext.height.toDouble()
        val fox = Angle.fromRadians((2 * PI) / 5)
        projectionMatrix = Matrix4.perspective(fox, aspect, 1.0, 100.0)
    }

    override suspend fun AutoClosableContext.render() {

        val transformationMatrix = getTransformationMatrix(
            frame / 100.0,
            projectionMatrix
        )
        device.queue.writeBuffer(
            uniformBuffer,
            0,
            transformationMatrix,
            0,
            transformationMatrix.size.toLong()
        )

        renderPassDescriptor = renderPassDescriptor.copy(
            colorAttachments = listOf(
                renderPassDescriptor.colorAttachments[0].copy(
                    view = renderingContext.getCurrentTexture()
                        .bind()
                        .createView()
                )
            )
        )

        val encoder = device.createCommandEncoder()
            .bind()

         encoder.beginRenderPass(renderPassDescriptor) {
            setPipeline(renderPipeline)
            setBindGroup(0, uniformBindGroup)
            setVertexBuffer(0, verticesBuffer)
            draw(Cube.cubeVertexCount)
            end()
        }

        val commandBuffer = encoder.finish()
            .bind()

        device.queue.submit(listOf(commandBuffer))


    }

}


private fun getTransformationMatrix(angle: Double, projectionMatrix: Matrix4): FloatArray {
    var viewMatrix = Matrix4.IDENTITY
    viewMatrix = viewMatrix.translated(0, 0, -4)

    viewMatrix = viewMatrix.rotated(
        Angle.fromRadians(Angle.fromRadians(angle).sine),
        Angle.fromRadians(Angle.fromRadians(angle).cosine),
        Angle.fromRadians(0)
    )

    return (projectionMatrix * viewMatrix).copyToColumns()
}