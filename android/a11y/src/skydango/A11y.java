package skydango;

import android.accessibilityservice.AccessibilityServiceInfo;
import android.app.UiAutomation;
import android.app.UiAutomationConnection;
import android.graphics.Rect;
import android.os.HandlerThread;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityNodeInfo;

import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.PrintStream;

/**
 * 读当前窗口的无障碍节点（文字 + 屏幕坐标），每个快照一行 JSON 打到 stdout。
 *
 * 和系统自带的 uiautomator 一样由 shell 用户经 app_process 起一个 UiAutomation 连接，
 * 只是不等 idle（游戏一直在渲染，uiautomator dump 永远等不到）。
 *
 * 用法：CLASSPATH=skydango-a11y.jar app_process /system/bin --nice-name=skydango-a11y skydango.A11y [dump | watch [间隔毫秒]]
 *   dump：打一行就退出
 *   watch：每隔一段时间看一次，节点变了（或隔了 1 秒心跳）就打一行；stdout 断了（adb 断开）就退出
 *
 * 一行的格式：{"t":设备时间毫秒,"pkg":"包名","nodes":[{"text":"…","desc":"…","cls":"TextView","id":"…","b":[左,上,右,下],"v":1}]}
 * 只列有文字或描述的节点；拿不到窗口时 pkg 为 null、nodes 为空。
 */
public final class A11y {
    private static final int FLAG_DONT_SUPPRESS_ACCESSIBILITY_SERVICES = 1;  // UiAutomation.connect(flags)
    private static final int MAX_DEPTH = 40;
    private static final long HEARTBEAT_MS = 1000;

    private static volatile long eventAt;  // 最近一次无障碍事件（System.nanoTime）

    /**
     * adb exec-out 没有单独的 stderr（设备端 stderr 并进 stdout），所以出错时往 stdout 打一行 "error: …"，
     * 不是 JSON，Python 端收进 error。最后一律 System.exit：别让还开着的线程把进程挂在设备上。
     */
    public static void main(String[] args) throws Exception {
        PrintStream out = new PrintStream(new FileOutputStream(FileDescriptor.out), true, "UTF-8");
        int code = 0;
        try {
            run(args, out);
        } catch (Throwable e) {
            out.println("error: " + e);
            code = 1;
        }
        out.flush();
        System.exit(code);
    }

    private static void run(String[] args, PrintStream out) throws Exception {
        String mode = args.length > 0 ? args[0] : "dump";
        long interval = args.length > 1 ? Long.parseLong(args[1]) : 200;

        HandlerThread thread = new HandlerThread("skydango-a11y");
        thread.start();
        UiAutomation ui = new UiAutomation(thread.getLooper(), new UiAutomationConnection());
        boolean connected = false;
        try {
            ui.connect(FLAG_DONT_SUPPRESS_ACCESSIBILITY_SERVICES);
            connected = true;
            AccessibilityServiceInfo info = ui.getServiceInfo();
            info.flags |= AccessibilityServiceInfo.FLAG_INCLUDE_NOT_IMPORTANT_VIEWS
                    | AccessibilityServiceInfo.FLAG_REPORT_VIEW_IDS;
            ui.setServiceInfo(info);
            // 刚连上时窗口信息还没同步过来，根节点是空的：最多等 2 秒
            for (int i = 0; i < 40 && ui.getRootInActiveWindow() == null; i++) {
                Thread.sleep(50);
            }
            if (!"watch".equals(mode)) {
                out.println(snapshot(ui));
                return;
            }
            ui.setOnAccessibilityEventListener(new UiAutomation.OnAccessibilityEventListener() {
                @Override
                public void onAccessibilityEvent(AccessibilityEvent event) {
                    eventAt = System.nanoTime();
                }
            });
            String last = null;
            long lastOut = 0;
            long lastRead = -1;
            while (true) {
                long now = System.nanoTime();
                boolean beat = now - lastOut >= HEARTBEAT_MS * 1_000_000L;
                if (eventAt != lastRead || beat) {  // 没有事件就不重新遍历
                    lastRead = eventAt;
                    String nodes = nodes(ui);
                    if (!nodes.equals(last) || beat) {
                        out.println("{\"t\":" + System.currentTimeMillis() + "," + nodes + "}");
                        last = nodes;
                        lastOut = now;
                    }
                }
                if (out.checkError()) {
                    break;  // adb 断开了
                }
                Thread.sleep(interval);
            }
        } finally {
            if (connected) {
                try {
                    ui.disconnect();
                } catch (Throwable ignored) {
                    // 连接已经断了；别盖掉原来的异常
                }
            }
            thread.quit();
        }
    }

    private static String snapshot(UiAutomation ui) {
        return "{\"t\":" + System.currentTimeMillis() + "," + nodes(ui) + "}";
    }

    /** "pkg":…,"nodes":[…]（不带外层花括号，方便比较有没有变）。 */
    private static String nodes(UiAutomation ui) {
        AccessibilityNodeInfo root = ui.getRootInActiveWindow();
        if (root == null) {
            return "\"pkg\":null,\"nodes\":[]";
        }
        StringBuilder sb = new StringBuilder();
        sb.append("\"pkg\":").append(str(root.getPackageName())).append(",\"nodes\":[");
        Rect rect = new Rect();
        int[] count = {0};
        walk(root, 0, sb, rect, count);
        sb.append(']');
        return sb.toString();
    }

    private static void walk(AccessibilityNodeInfo node, int depth, StringBuilder sb, Rect rect, int[] count) {
        CharSequence text = node.getText();
        CharSequence desc = node.getContentDescription();
        if (notEmpty(text) || notEmpty(desc)) {
            node.getBoundsInScreen(rect);
            if (count[0]++ > 0) {
                sb.append(',');
            }
            String cls = node.getClassName() == null ? null : node.getClassName().toString();
            if (cls != null && cls.lastIndexOf('.') >= 0) {
                cls = cls.substring(cls.lastIndexOf('.') + 1);
            }
            String id = node.getViewIdResourceName();
            if (id != null && id.indexOf('/') >= 0) {
                id = id.substring(id.indexOf('/') + 1);
            }
            sb.append("{\"text\":").append(str(text))
                    .append(",\"desc\":").append(str(desc))
                    .append(",\"cls\":").append(str(cls))
                    .append(",\"id\":").append(str(id))
                    .append(",\"b\":[").append(rect.left).append(',').append(rect.top).append(',')
                    .append(rect.right).append(',').append(rect.bottom).append(']')
                    .append(",\"v\":").append(node.isVisibleToUser() ? 1 : 0)
                    .append('}');
        }
        if (depth < MAX_DEPTH) {
            int n = node.getChildCount();
            for (int i = 0; i < n; i++) {
                AccessibilityNodeInfo child = node.getChild(i);
                if (child != null) {
                    walk(child, depth + 1, sb, rect, count);
                    child.recycle();
                }
            }
        }
    }

    private static boolean notEmpty(CharSequence s) {
        return s != null && s.length() > 0;
    }

    private static String str(CharSequence s) {
        if (s == null) {
            return "null";
        }
        StringBuilder sb = new StringBuilder(s.length() + 2).append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '"' || c == '\\') {
                sb.append('\\').append(c);
            } else if (c < 0x20) {
                sb.append(String.format("\\u%04x", (int) c));
            } else {
                sb.append(c);
            }
        }
        return sb.append('"').toString();
    }
}
