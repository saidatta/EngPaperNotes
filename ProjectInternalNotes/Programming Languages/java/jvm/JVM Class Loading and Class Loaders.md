https://juejin.cn/post/7246777406386782265

#### Overview
Class loaders play a crucial role in the Java Virtual Machine (JVM) by loading Java classes into memory, converting them into Java objects, and ensuring that Java applications run smoothly. They are responsible for the dynamic class loading mechanism in Java, allowing Java applications to load classes at runtime just before they are required for execution.
![[Screenshot 2024-04-24 at 7.59.13 PM.png]]
#### Types of Class Loaders
1. **Bootstrap Class Loader**:
   - Loads the core Java APIs located in the `JAVA_HOME/jre/lib` directory.
   - Implemented natively in C++ as part of the JVM.
   - Does not appear as a Java object, hence not accessible from Java applications.

2. **Extension Class Loader**:
   - Loads classes from the extensions directories (`JAVA_HOME/jre/lib/ext`).
   - Written in Java and delegates to the Bootstrap loader.

3. **Application Class Loader**:
   - Loads classes found on the Java classpath, which typically includes class files from the directories or JAR files specified by the classpath environment variable.
   - Also written in Java, and it delegates to the Extension Class Loader.

4. **Custom Class Loader**:
   - Allows developers to extend the ClassLoader class to alter the JVM's class loading mechanism by overriding methods like `findClass()`.
   - Useful for loading classes from non-standard sources such as a network or for implementing classes with custom byte-code transformations.
![[Screenshot 2024-04-24 at 7.59.57 PM.png]]
#### Class Loading Mechanism
Class loading in the JVM consists of three main activities: loading, linking, and initialization.
1. **Loading**:
   - The class loader reads the `.class` file (bytecode), generates a corresponding `Class` object in the heap, and then goes on to linking.
2. **Linking**:
   - Consists of verification (ensures bytecode integrity), preparation (allocates memory for class variables and initializes them), and resolution (transforms symbolic references from the type into direct references).
3. **Initialization**:
   - Involves executing static initializers and the static initialization blocks of a class and its parents in an upward recursive manner.
#### The Parent Delegation Model
The parent delegation model is the default mechanism for class loading in Java:
- When a class loader is asked to load a class, it delegates the task to its parent class loader.
- This continues until the request reaches the top of the hierarchy (Bootstrap Class Loader).
- If the parent class loaders do not find the class, the class loader that originally received the request attempts to load the class itself.
- Benefits include prevention of class duplication, enhanced security (prevents user-defined classes from overriding core Java classes), and namespace management.
#### Custom Class Loader Example
```java
public class CustomClassLoader extends ClassLoader {
    private Map<String, Class<?>> classes = new HashMap<>();
    private String classPath;
    private Set<String> allowedPackages;

    public CustomClassLoader(String classPath, Set<String> allowedPackages) {
        this.classPath = classPath;
        this.allowedPackages = allowedPackages;
    }

    @Override
    public Class<?> loadClass(String name) throws ClassNotFoundException {
        Class<?> clazz = classes.get(name);
		if (clazz != null) { 
			return clazz; 
		} 
		if (!isAllowed(name)) { 
			throw new ClassNotFoundException("Class " + name + " not allowed"); 
		}


        if (clazz == null && allowedPackages.stream().anyMatch(name::startsWith)) {
            byte[] bytes = getClassData(name);
            if (bytes != null) {
                clazz = defineClass(name, bytes, 0, bytes.length);
                classes.put(name, clazz);
            }
        }
        return super.loadClass(name, true);
    }

	private boolean isAllowed(String name) {
		for (String allowedPackage : allowedPackages) { 
			if (name.startsWith(allowedPackage)) { 
				return true; 
			} 
		} 
		return false; 
	}

    private byte[] getClassData(String name) {
        try {
            InputStream is = new FileInputStream(new File(classPath, name.replace('.', '/') + ".class"));
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            byte[] buffer = new byte[4096];
            int bytesread;
            while ((bytesread = is.read(buffer)) != -1) {
                baos.write(buffer, 0, bytesread);
            }
            return baos.toByteArray();
        } catch (IOException e) {
            return null;
        }
    }
}
```
#### Usage
- This `CustomClassLoader` allows loading of `.class` files from a specified directory and only from allowed packages. This helps in providing a sandbox environment, useful in multi-application hosting scenarios where classes from one application need to be isolated from another.
#### Summary
Class loaders are an integral part of JVM architecture, enabling dynamic class loading, isolation, and delegation, which are critical for Java application performance and security. Custom class loaders offer flexibility, enhancing security and enabling dynamic behavior modifications.