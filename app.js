document.getElementById('fileInput').addEventListener('change', async (event) => {
    const files = event.target.files; // 获取用户选择的文件列表
    const statusElement = document.getElementById('status');

    if (!files.length) {
        statusElement.textContent = '未选择任何文件！';
        return;
    }

    statusElement.textContent = `正在处理 ${files.length} 张图片...`;

    try {
        // 遍历文件并将其转换为 Base64
        const base64Images = await Promise.all(
            Array.from(files).map(async (file) => {
                const base64 = await fileToBase64(file);
                return { name: file.name, data: base64 };
            })
        );

        // 发送 Base64 数据到后端
        statusElement.textContent = '正在上传图片到服务器...';
        await sendImagesToBackend(base64Images);

        statusElement.textContent = '所有图片已成功上传并开始处理！';
    } catch (error) {
        console.error('上传过程中出现错误:', error);
        statusElement.textContent = '上传失败，请重试！';
    }
});

// 将文件转换为 Base64
function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result.split(',')[1]); // 去掉 Base64 的前缀
        reader.onerror = (error) => reject(error);
        reader.readAsDataURL(file);
    });
}

// 发送 Base64 图片数据到后端
async function sendImagesToBackend(images) {
    const backendUrl = 'http://localhost:5000/upload'; // 替换为您的后端 API 地址

    const response = await fetch(backendUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ images }),
    });

    if (!response.ok) {
        throw new Error(`HTTP 错误！状态码: ${response.status}`);
    }

    const result = await response.json();
    console.log('后端响应:', result);
}