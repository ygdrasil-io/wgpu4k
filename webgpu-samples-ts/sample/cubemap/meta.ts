export default {
    name: 'Cubemap',
    description:
        'This example shows how to render and sample from a cubemap texture.',
    filename: __DIRNAME__,
    sources: [
        {path: 'main.ts'},
        {path: '../../shaders/basic.vert.wgsl'},
        {path: './sampleCubemap.frag.wgsl'},
        {path: '../../meshes/cube.ts'},
    ],
};
