## 一、先把项目拉到自己电脑

```bash
git clone https://github.com/Dickey007s/Fuzzy_matching_2026.git
cd Fuzzy_matching_2026
```

---

## 二、查看所有分支（5 个模块都在）

```bash
git fetch
git branch -a
```

你会看到：

- module1

- module2

- module3

- module4

- module5

- main

---

## 三、切换到你的分支

- 成员 x → **modulex**

如：
```bash
git switch module2
```

---

## 四、日常工作

1. **写代码**（只改你自己模块的文件）

2. 保存修改

```bash
git add .
```

3. 提交（说明你干了啥）

```bash
git commit -m "module2: 完成CKKS加密"
```

4. 推送到 GitHub

```bash
git push origin module2
```

---

## 五、需要用到别人更新的文件怎么办？

### 情况 A：需要引用已完成且合并到main分支的代码

**做法：**
```bash
git pull origin main
```

就能把最新的公共代码同步到你的分支。


### 情况 B：需要实时用别人最新代码

**例如：成员4负责聚类优化，正在开发 `kmeans_cluster.py` 文件，还没写完，暂时不想合并到 main（避免影响main的稳定性），但成员3需要用他刚写好的 `build_column_matrix` 函数来处理列矩阵数据，必须用他最新的代码。**

**做法：**

1. 成员4提交推送到自己分支
```bash
git add . 
git commit -m "module4: 完成build_column_matrix函数开发"
git push origin module4
```
2. 成员3执行：
```bash
git fetch
git merge origin/moduleX
```

就能把对方分支的最新内容合并到你的分支。

---

## 六、最后合并

所有人写完后，组长执行：

```bash
git switch main
git merge module1
git merge module2
git merge module3
git merge module4
git merge module5
git push origin main
```

✅ main 就是完整项目！