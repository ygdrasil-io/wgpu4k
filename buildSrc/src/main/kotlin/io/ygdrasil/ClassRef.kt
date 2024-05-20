package io.ygdrasil

import com.squareup.kotlinpoet.ClassName
import com.squareup.kotlinpoet.ParameterSpec

sealed class TypeRef(val packageName: String, val className: String) {

    abstract val canonicalName: String

    fun toClassName(): ClassName = ClassName(packageName, className)

}

class GenericClassRef(packageName: String, className: String, val genericType: ClassRef
) : TypeRef(packageName, className) {
    override val canonicalName: String = "$packageName.$className<${genericType.canonicalName}>"
}

class ClassRef(packageName: String, className: String) : TypeRef(packageName, className) {
    override val canonicalName: String = "$packageName.$className"
}
