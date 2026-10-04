// 只用来编译的空壳，签名照 Android 12（API 32；构造函数、connect、disconnect 是隐藏 API）；不打进 jar，运行时用系统里真的类。
package android.app;

import android.accessibilityservice.AccessibilityServiceInfo;
import android.os.Looper;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityNodeInfo;

public final class UiAutomation {
    public interface OnAccessibilityEventListener {
        void onAccessibilityEvent(AccessibilityEvent event);
    }

    public UiAutomation(Looper looper, IUiAutomationConnection connection) {
        throw new RuntimeException("stub");
    }

    public void connect(int flags) {
        throw new RuntimeException("stub");
    }

    public void disconnect() {
        throw new RuntimeException("stub");
    }

    public AccessibilityServiceInfo getServiceInfo() {
        throw new RuntimeException("stub");
    }

    public void setServiceInfo(AccessibilityServiceInfo info) {
        throw new RuntimeException("stub");
    }

    public void setOnAccessibilityEventListener(OnAccessibilityEventListener listener) {
        throw new RuntimeException("stub");
    }

    public AccessibilityNodeInfo getRootInActiveWindow() {
        throw new RuntimeException("stub");
    }
}
