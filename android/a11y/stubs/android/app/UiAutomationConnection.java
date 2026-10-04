// 只用来编译的空壳（隐藏 API），不打进 jar，运行时用系统里真的类。
package android.app;

public final class UiAutomationConnection implements IUiAutomationConnection {
    public UiAutomationConnection() {
        throw new RuntimeException("stub");
    }
}
