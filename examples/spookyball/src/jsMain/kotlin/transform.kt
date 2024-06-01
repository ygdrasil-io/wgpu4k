import org.khronos.webgl.ArrayBuffer
import org.khronos.webgl.Float32Array

val DEFAULT_ORIENTATION = Float32Array(arrayOf(0f, 0f, 0f, 0f))

@OptIn(ExperimentalJsExport::class)
@JsExport
val DEFAULT_POSITION = MyVector3(0.0, 0.0, 0.0)

@OptIn(ExperimentalJsExport::class)
@JsExport
val DEFAULT_SCALE = MyVector3(1.0, 1.0, 1.0)

@OptIn(ExperimentalJsExport::class)
@JsExport
class TransformKt(options: dynamic) {

    var position: Float32Array
    var orientation: Float32Array
    var scale: Float32Array
    var localMatrix: Float32Array
    var worldMatrix: Float32Array
    var localMatrixDirty = true
    var worldMatrixDirty = true

    init {

        var buffer: ArrayBuffer
        var offset = 0

        if (options.externalStorage != null) {
            buffer = options.externalStorage.buffer
            offset = options.externalStorage.offset
        } else {
            buffer = Float32Array(42).buffer
        }

        position = Float32Array(buffer, offset, 3)
        orientation = Float32Array(buffer, offset + 3 * Float32Array.BYTES_PER_ELEMENT, 4)
        scale = Float32Array(buffer, offset + 7 * Float32Array.BYTES_PER_ELEMENT, 3)
        localMatrix = Float32Array(buffer, offset + 10 * Float32Array.BYTES_PER_ELEMENT, 16)
        worldMatrix = Float32Array(buffer, offset + 26 * Float32Array.BYTES_PER_ELEMENT, 16)

        if (options.transform) {
            val storage = Float32Array(position.buffer, position.byteOffset, 42)
            storage.set(Float32Array(options.transform.actual.position.buffer, options.transform.actual.position.byteOffset, 42))
            localMatrixDirty = options.transform.localMatrixDirty;
        } else {
            if (options.position) {
                position.set(options.position as Array<Float>)
            }
            orientation.set(if(options.orientation) options.orientation as Float32Array else DEFAULT_ORIENTATION)
            scale.set(if(options.scale) options.scale as Float32Array else DEFAULT_SCALE.toJS32Array())
        }

        if (options.parent) {
            options.parent.addChild(this);
        }
    }

}