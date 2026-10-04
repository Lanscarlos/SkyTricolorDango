// 只用来编译的空壳，签名照 Android 12（API 32）；不打进 jar，运行时用系统里真的类。
// 常量会被 javac 内联进 A11y.class，值必须和系统一致。
package android.accessibilityservice;

public class AccessibilityServiceInfo {
    public static final int FLAG_INCLUDE_NOT_IMPORTANT_VIEWS = 0x00000002;
    public static final int FLAG_REPORT_VIEW_IDS = 0x00000010;
    public int flags;
}
