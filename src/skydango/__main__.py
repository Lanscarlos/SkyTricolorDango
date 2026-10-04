from .cli import main

if __name__ == "__main__":  # Windows 上 ultralytics 的 dataloader 多进程（spawn）会重新导入主模块，不加这个会再跑一遍命令
    main()
