#!/bin/bash

# Phi项目GitHub上传脚本
# 请在运行前替换相关信息

echo "🚀 Phi项目GitHub上传指南"
echo "=============================="

echo "第1步：配置Git用户信息"
echo "请运行以下命令（替换为你的真实信息）："
echo "git config user.email 'your-github-email@example.com'"
echo "git config user.name 'Your GitHub Username'"
echo ""

echo "第2步：更新远程仓库地址"
echo "请运行以下命令（替换YOUR_USERNAME为你的GitHub用户名）："
echo "git remote set-url origin https://github.com/YOUR_USERNAME/Phi.git"
echo ""

echo "第3步：推送到GitHub"
echo "git push -u origin main"
echo ""

echo "第4步：（可选）推送图片数据"
echo "由于图片文件较大，建议："
echo "1. 使用Git LFS: git lfs track '*.jpg' && git add .gitattributes"
echo "2. 或将数据集单独上传到HuggingFace Hub"
echo ""

echo "第5步：验证上传"
echo "访问 https://github.com/YOUR_USERNAME/Phi 查看结果"
echo ""

echo "📊 当前项目统计："
echo "- 主要脚本：$(find examples/scripts -name '*.py' | wc -l) 个"
echo "- 数据集：$(find data -name '*.csv' | wc -l) 个CSV文件"
echo "- 文档：README.md, CONTRIBUTING.md, LICENSE"
echo "- GPT评估脚本：2 个"
echo ""

echo "✅ 项目已准备就绪，可以上传到GitHub！"
