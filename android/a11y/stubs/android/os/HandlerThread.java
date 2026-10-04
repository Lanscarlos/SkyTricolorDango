// 只用来编译的空壳，不打进 jar，运行时用系统里真的类。
package android.os;

public class HandlerThread extends Thread {
    public HandlerThread(String name) {
        throw new RuntimeException("stub");
    }

    public Looper getLooper() {
        throw new RuntimeException("stub");
    }

    public boolean quit() {
        throw new RuntimeException("stub");
    }
}
