// 只用来编译的空壳，不打进 jar，运行时用系统里真的类。
package android.view.accessibility;

import android.graphics.Rect;

public class AccessibilityNodeInfo {
    public CharSequence getText() { throw new RuntimeException("stub"); }
    public CharSequence getContentDescription() { throw new RuntimeException("stub"); }
    public CharSequence getClassName() { throw new RuntimeException("stub"); }
    public CharSequence getPackageName() { throw new RuntimeException("stub"); }
    public String getViewIdResourceName() { throw new RuntimeException("stub"); }
    public void getBoundsInScreen(Rect outBounds) { throw new RuntimeException("stub"); }
    public boolean isVisibleToUser() { throw new RuntimeException("stub"); }
    public int getChildCount() { throw new RuntimeException("stub"); }
    public AccessibilityNodeInfo getChild(int index) { throw new RuntimeException("stub"); }
    public void recycle() { throw new RuntimeException("stub"); }
}
