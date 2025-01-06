https://www.youtube.com/watch?v=KvRrrYiIxVI&t=1127s

**Audience:**  
These notes are intended for experienced Windows engineers familiar with user-mode development, threads, and the Windows API. We will dive deep into the steps for creating a window, how the process, thread, and window manager interact, and touch on relevant internals such as window classes, message queues, and the NT architecture behind the scenes.

---
## Overview
In Win32, windows (GUI elements like application main windows, dialog boxes, child controls) are tightly integrated with threads. A thread that creates a window:
- Owns the window’s message queue.
- Processes messages in a loop (the "message pump").
- Must remain responsive to keep the UI active, otherwise the window appears frozen ("Not Responding").
- 
To create a window, you typically:
1. **Register a Window Class:**  
   Defines properties for a type of window, including its window procedure (WndProc), background color, icon, cursor, etc.
2. **Create an Instance (Window):**  
   Using `CreateWindowEx` or `CreateWindow` with the registered class name.
3. **Show and Update the Window:**  
   Make the window visible and paint its client area.
4. **Message Loop:**  
   Run a loop calling `GetMessage`, `TranslateMessage`, `DispatchMessage` to handle incoming events (mouse clicks, keyboard input, redraw requests, etc.).

---
## Processes, Threads, and Windows
- **Process and Threads:**
  A process can have multiple threads. Typically, a single GUI thread manages all top-level windows. Technically, different threads can own different windows.
- **Message Queues:**
  When a thread creates (or requests) a window, Windows assigns a message queue to that thread (if not already assigned). All window messages destined for that thread’s windows go into this queue.
- **Responsiveness:**
  The thread owning the window must periodically call `GetMessage` (or `PeekMessage`) and `DispatchMessage` to process incoming messages. Without this, the window won't respond to user input and system events.

---
## Internals of Window Classes
**Window Class (WNDCLASS or WNDCLASSEX):**
- The **window class** template describes a type of window. Once registered with `RegisterClass` or `RegisterClassEx`, the system stores:
  - A pointer to the window procedure (a callback function).
  - Default icons, cursors, background brushes.
  - Style flags that affect redrawing and other behaviors.

**Window Procedure:**
- A function with the signature:
  ```c
  LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
  ```
  This function receives messages like `WM_PAINT`, `WM_CLOSE`, and must return appropriate values. Unhandled messages often get passed to `DefWindowProc` for default handling.

**Object Internals:**
- Internally, USER32/GDI32 and the window manager maintain data structures called `WND` and `CLASS` objects in kernel memory managed by the Window Manager (Win32k.sys in kernel mode).
- `RegisterClass` or `RegisterClassEx` calls `NtUserRegisterClass` (undocumented) internally, creating a kernel-mode structure representing the class.
- The handle to a window (HWND) is essentially an index into internal tables that reference these kernel-mode structures.

---
## Step-by-Step Example

### 1. Registering a Window Class

```c
#include <windows.h>

LRESULT CALLBACK MyWndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hWnd, &ps);
        // Get client rectangle
        RECT rc; GetClientRect(hWnd, &rc);
        // Draw an ellipse
        Ellipse(hdc, rc.left, rc.top, rc.right, rc.bottom);
        EndPaint(hWnd, &ps);
        return 0;
    }
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    default:
        return DefWindowProc(hWnd, uMsg, wParam, lParam);
    }
}
```

This `MyWndProc` handles painting (WM_PAINT) and close/destroy (WM_DESTROY).

**Register the class:**
```c
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow) {
    WNDCLASS wc = {0};
    wc.style = CS_HREDRAW | CS_VREDRAW; // Redraw on horizontal/vertical resize
    wc.lpfnWndProc = MyWndProc;          // Window procedure
    wc.hInstance = hInstance;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hIcon   = LoadIcon(NULL, IDI_APPLICATION);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW+1);
    wc.lpszClassName = L"MyWinClass";

    if (!RegisterClass(&wc)) {
        return 0;
    }
```

`RegisterClass` returns an atom (integer) if successful, which uniquely identifies the class in this process.

---
### 2. Creating the Window

```c
    HWND hWnd = CreateWindow(
        L"MyWinClass",           // Class name
        L"My Application",       // Window title
        WS_OVERLAPPEDWINDOW,     // Style (title bar, resizable, etc.)
        CW_USEDEFAULT, CW_USEDEFAULT,  // Position
        CW_USEDEFAULT, CW_USEDEFAULT,  // Size
        NULL,                    // No parent (top-level window)
        NULL,                    // No menu
        hInstance,
        NULL                     // No extra creation data
    );

    if (!hWnd) {
        return 0;
    }

    ShowWindow(hWnd, nCmdShow);
    UpdateWindow(hWnd);
```

**Internals:**
- `CreateWindow` calls `NtUserCreateWindowEx` internally, which creates kernel-mode structures. The `WndProc` address is known from the class; the system sets up the window’s internal data structures.
- The thread that calls `CreateWindow` now owns this window and must run a message loop.

---
### 3. The Message Loop

```c
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);   // Translate keyboard input to character messages
        DispatchMessage(&msg);    // Call WndProc for the target window
    }

    return (int)msg.wParam;
}
```

**Internals:**
- `GetMessage` retrieves messages from the thread’s message queue.
- The queue is populated by the window manager with events from the user (mouse moves, clicks), system (resize, paint), or application (timers, posted messages).
- `DispatchMessage` uses the HWND in `msg.hwnd` to find the associated window procedure and calls it.

---
## Running and Observing the Window
When running the above code:
- You see a main window with a white background and an ellipse drawn in its client area.
- Resizing the window triggers `WM_PAINT` again, causing the ellipse to redraw.
- Clicking the close (X) button sends `WM_CLOSE` and eventually `WM_DESTROY`. `PostQuitMessage(0)` posts `WM_QUIT`, causing `GetMessage` to return 0 and the loop to end, closing the application.
---
## Tools and Internals Observation
1. **WinSpy or Spy++:**
   - These tools enumerate all top-level windows and their hierarchy.
   - You can see your window’s class name, styles, parent/child relationships.
   - Internally, each window belongs to a thread, and the tool can show which thread owns which windows.
2. **Kernel Debugger (WinDbg):**
   - At kernel-level, windows belong to internal data structures managed by `win32k.sys`.
   - `!wnd` command (in older versions of WinDbg with the right extensions) can dump kernel-mode window structures.
   - The `EPROCESS`, `ETHREAD`, and `W32THREAD` structures connect threads to their message queues and windows.

3. **Message Internals:**
   - Messages are identified by constants like `WM_PAINT (0x000F)`, `WM_DESTROY (0x0002)`, `WM_QUIT (0x0012)`.
   - Most standard UI handling (like handling close box, resizing) is done by default in `DefWindowProc`.

---
## Handling More Complexity
- **Multiple Windows:**
  You can create multiple windows with the same class. Each will receive its own messages. As long as they belong to the same thread, they share the same message queue.
- **Modal Dialogs:**
  Modal loops temporarily alter the message processing, but the concept remains: dispatch messages and respond in the window procedure.
- **Custom Drawing and GDI+ or Direct2D:**
  Instead of `Ellipse`, you can use more advanced graphics APIs. The process is similar: handle `WM_PAINT`, get `HDC` from `BeginPaint`, draw, then `EndPaint`.

---
## Tips and Best Practices
- **Use `RegisterClassEx` for more icon/size parameters if needed.**
- **Always handle `WM_DESTROY`:**  
  Post a quit message for the main window to terminate the loop gracefully.
- **Minimize blocking operations in `WndProc`:**
  Keep UI responsive by avoiding long operations. Instead, post messages or run work on other threads.
- **UNICODE Support:**
  Use the wide-character versions of APIs (`wWinMain`, `RegisterClassW`, etc.) for full Unicode support.

---
## Summary
Creating a window in Windows involves:
1. Defining and registering a window class with a `WndProc`.
2. Creating a window instance with `CreateWindow` or `CreateWindowEx`.
3. Showing and updating the window to make it visible.
4. Running a message loop to process events.
Under the hood, this involves calling into native user and GDI subsystems, allocating kernel-mode objects, and setting up a message queue. Tools like WinSpy++ and kernel debugging commands can reveal these internals. The final result is a responsive, interactive GUI element controlled by your application’s thread.
