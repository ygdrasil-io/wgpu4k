package darwin 

import org.rococoa.ObjCClass
import org.rococoa.Rococoa

/**
 * This file was autogenerated by [JNAerator](http://jnaerator.googlecode.com/),<br></br>
 * a tool written by [Olivier Chafik](http://ochafik.free.fr/) that [uses a few opensource projects.](http://code.google.com/p/jnaerator/wiki/CreditsAndLicense).<br></br>
 * For help, please visit [NativeLibs4Java](http://nativelibs4java.googlecode.com/), [Rococoa](http://rococoa.dev.java.net/), or [JNA](http://jna.dev.java.net/).
 */
abstract class NSAppleScript : NSObject() {
	interface _Class : ObjCClass {
		fun alloc(): NSAppleScript
	}

	/**
	 * Given a URL that locates a script, in either text or compiled form, initialize.  Return nil and a pointer to an error information dictionary if an error occurs.  This is a designated initializer for this class.<br></br>
	 * Given a URL that locates a script, in either text or compiled form, initialize.  Return nil and a pointer to an error information dictionary if an error occurs.  This is a designated initializer for this class.<br></br>
	 * Given a URL that locates a script, in either text or compiled form, initialize.  Return nil and a pointer to an error information dictionary if an error occurs.  This is a designated initializer for this class.<br></br>
	 * Original signature : `-(id)initWithContentsOfURL:(NSURL*) error:(NSDictionary**)`<br></br>
	 * *native declaration : line 28*
	 */
	abstract fun initWithContentsOfURL_error(url: NSURL?, errorInfo: com.sun.jna.ptr.ByReference?): NSAppleScript?

	/**
	 * Given a string containing the AppleScript source code of a script, initialize.  Return nil if an error occurs.  This is also a designated initializer for this class.<br></br>
	 * Original signature : `-(id)initWithSource:(NSString*)`<br></br>
	 * *native declaration : line 31*
	 */
	abstract fun initWithSource(source: String?): NSAppleScript?

	/**
	 * Return the source code of the script if it is available, nil otherwise.  It is possible for an NSAppleScript that has been instantiated with -initWithContentsOfURL:error: to be a script for which the source code is not available, but is nonetheless executable.<br></br>
	 * Original signature : `-(NSString*)source`<br></br>
	 * *native declaration : line 34*
	 */
	abstract fun source(): String?

	/**
	 * Return yes if the script is already compiled, no otherwise.<br></br>
	 * Original signature : `-(BOOL)isCompiled`<br></br>
	 * *native declaration : line 37*
	 */
	abstract fun isCompiled(): Boolean

	/**
	 * Compile the script, if it is not already compiled.  Return yes for success or if the script was already compiled, no and a pointer to an error information dictionary otherwise.<br></br>
	 * Original signature : `-(BOOL)compileAndReturnError:(NSDictionary**)`<br></br>
	 * *native declaration : line 40*
	 */
	abstract fun compileAndReturnError(errorInfo: com.sun.jna.ptr.ByReference?): Boolean

	/**
	 * Execute the script, compiling it first if it is not already compiled.  Return the result of executing the script, or nil and a pointer to an error information dictionary for failure.<br></br>
	 * Original signature : `-(NSAppleEventDescriptor*)executeAndReturnError:(NSDictionary**)`<br></br>
	 * *native declaration : line 43*
	 */
	abstract fun executeAndReturnError(errorInfo: com.sun.jna.ptr.ByReference?): NSAppleEventDescriptor?

	/**
	 * Execute an Apple event in the context of the script, compiling the script first if it is not already compiled.  Return the result of executing the event, or nil and a pointer to an error information dictionary for failure.<br></br>
	 * Original signature : `-(NSAppleEventDescriptor*)executeAppleEvent:(NSAppleEventDescriptor*) error:(NSDictionary**)`<br></br>
	 * *native declaration : line 46*
	 */
	abstract fun executeAppleEvent_error(
		event: NSAppleEventDescriptor?,
		errorInfo: com.sun.jna.ptr.ByReference?
	): NSAppleEventDescriptor?

	companion object {
		private val CLASS: _Class = Rococoa.createClass("NSAppleScript", _Class::class.java)

		/**
		 * Factory method<br></br>
		 *
		 * @see .initWithContentsOfURL_error
		 */
		fun createWithContentsOfURL_error(url: NSURL?, errorInfo: com.sun.jna.ptr.ByReference?): NSAppleScript? {
			return CLASS.alloc().initWithContentsOfURL_error(url, errorInfo)
		}

		/**
		 * Factory method<br></br>
		 *
		 * @see .initWithSource
		 */
		fun createWithSource(source: String?): NSAppleScript? {
			return CLASS.alloc().initWithSource(source)
		}

		fun alloc(): NSAppleScript? {
			return CLASS.alloc()
		}
	}
}
