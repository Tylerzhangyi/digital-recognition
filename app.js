async function uploadImages() {
    const files = document.getElementById('imageInput').files;
    if (!files.length) {
        alert('请先选择图片');
        return;
    }

    const images = Array.from(files).map(file => ({
        name: file.name,
        data: await fileToBase64(file)
    }));

    try {
        const backendUrl = 'http://<public-ip>:5001/upload';
        const response = await fetch(backendUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ images })
        });

        if (!response.ok) {
            throw new Error(`HTTP 错误！状态码: ${response.status}`);
        }

        const result = await response.json();
        document.getElementById('status').textContent = '图片已成功上传并排队处理！';

        // 查询结果
        for (const image of images) {
            await checkResult(image.name);
        }
    } catch (error) {
        console.error('上传过程中出现错误:', error);
        document.getElementById('status').textContent = '上传失败，请重试！';
    }
}

function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result.split(',')[1]); // 去掉 Base64 的前缀
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

async function checkResult(imageName) {
    const backendUrl = `http://<public-ip>:5001/get_result_url?image_name=${imageName}`;
    const response = await fetch(backendUrl);
    const data = await response.json();

    if (data.url) {
        document.getElementById('status').innerHTML += `<br><a href="${data.url}" target="_blank">${imageName} 结果</a>`;
    } else {
        console.error(`无法获取 ${imageName} 的结果`);
    }
}