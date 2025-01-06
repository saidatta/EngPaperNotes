# Explanation of "use ZBID_IMMERSIVE_NOTIFICATION instead of ZBID_UIACCESS to prevent fps cut in half"

**Context:**
This statement likely appears in discussions about creating overlays or windows on top of full-screen applications or games on Windows. When developers create custom overlays (for example, an in-game FPS counter, a streaming overlay, or some UI element on top of a running 3D application), they must choose how the overlay window is positioned and integrated with the desktop and the rendering pipeline.

**Key Terms:**
- **ZBID_UIACCESS** and **ZBID_IMMERSIVE_NOTIFICATION**: These refer to **Z-order band IDs** used internally by Windows to place windows in certain positions in the z-order (the stacking order of windows on the screen). Different Z-order bands have different privileges and behaviors.

- **ZBID_UIACCESS**: Historically, this band might be used for windows that require higher privileges, or UI elements that need to appear above other windows. However, using it in a gaming context (like an overlay above a full-screen DirectX application) can interfere with certain system optimizations. The Windows compositor or the graphics pipeline might switch the game out of an exclusive full-screen mode or alter rendering paths, resulting in decreased performance, such as halving the game’s frame rate.

- **ZBID_IMMERSIVE_NOTIFICATION**: This band is typically used for notifications or Windows UI that is designed to appear above immersive applications (like full-screen apps or Windows 8/10 modern apps). Using this band allows you to place your overlay above the game without triggering some of the negative side effects associated with `ZBID_UIACCESS`. Essentially, it better aligns with how modern, overlay-style UI is handled, such as the Xbox Game Bar overlay. The Game Bar overlay is known to appear over games without drastically affecting performance. It likely uses this immersive notification z-band to achieve a performant overlay that doesn't break fullscreen optimizations.

**What Happens Internally:**
- When a game runs in exclusive full-screen mode, it may have direct front-buffer access or use exclusive swapchain optimizations. Introducing a high-privilege or incorrectly placed window (like one in the `ZBID_UIACCESS` band) can cause Windows to disable these optimizations. The game may drop out of exclusive full-screen mode into a windowed or borderless mode, often resulting in lower performance or a halved framerate.
  
- By using `ZBID_IMMERSIVE_NOTIFICATION`, Windows treats your overlay similar to how it treats system overlays designed for full-screen scenarios (like Game Bar). Thus, it does not disrupt the game’s rendering optimizations, preserving higher FPS.

**Practical Meaning for Developers:**
- If you are developing an overlay that needs to sit on top of a game, choosing the correct z-order band is crucial. Instead of using `ZBID_UIACCESS`, which might have seemed logical for a topmost privileged window, switch to `ZBID_IMMERSIVE_NOTIFICATION`. This ensures your overlay won’t cause the system to disable certain fullscreen optimizations, allowing the game to maintain a higher framerate.

**In Short:**
"Use `ZBID_IMMERSIVE_NOTIFICATION` instead of `ZBID_UIACCESS`" means:  
When creating an overlay window for a game or a full-screen app, placing it in the immersive notification z-band prevents the unwanted side effect of halving the game’s FPS, mimicking how the Xbox Game Bar or other system overlays maintain performance while still drawing on top of the application.
