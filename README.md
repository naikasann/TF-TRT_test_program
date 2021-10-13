# TF-TRT model test

## 概要

TensorflowからモデルをロードしてTF-TRTに変換・推論をするためのプログラム
TF-TRTについては- [Accelerating Inference In TF-TRT User Guide :: NVIDIA Deep Learning Frameworks Documentation](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html)を参照。
これやreference pageをみて自分好みに使いやすいようにいじった感じ。

`main.py`はresnetをkeras.applicationから呼び出しTensorRTに変換。その後np.random.uniformを推論してみるプログラムが入っている。
tftrtmodelのクラスの使い方は中身のコメント文を見てもらえればわかると思う。

`main.py`の実行方法はそれぞれ下記の環境構築後

``` bash
    python main.py
```

で実行できるはず。ただそんないいプログラムじゃないので書き換え、自分の用途に換装が必要。
それを簡単に行うためのプログラム。

---

## python library

Pythonのライブラリは`requiresment.txt`を見てもらえればdockerの環境ですが必要なものを入れてる感じなので。
Jetsonでのビルドの場合はまた別。（記載予定なし。）

---

## Jetson

Jetson環境ではJetpack4.6を用いてテストを行い動作確認済み。
Jetpack 4.6は[JetPack SDK | NVIDIA Developer](https://developer.nvidia.com/embedded/jetpack)を参照

### Jetson 環境

- L4T 32.6.1
- NVIDIA CUDA 10.2
- NVIDIA cuDNN 8.2.1
- TensorRT 8.0.1

---

## docker

環境構築にはdocker(NVIDIAの提供しているNGC)を用いて環境構築を行った
docker内の環境は（[TensorRT | NVIDIA NGC](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt)参照）

### Docker 環境

- Ubuntu 20.04
- NVIDIA CUDA 11.3.1
- NVIDIA cuDNN 8.2.1
- TensorRT 7.2.3.4

dockerの環境構築手順は

``` bash
    docker pull nvcr.io/nvidia/tensorrt:21.06-py3
```

を行い。その後`Dockerfile`があるディレクトリで

```bash
    docker image build -t <imagename:tag> .
```

を行いビルドする。dockerのlunchは

``` bash
    docker container run \
    --rm --gpus "device=0" -it \
    -e LOCAL_UID=$(id -u $USER) -e LOCAL_GID=$(id -g $USER) \
    <imagename:tag> bash
```

でlunchする。

---

## Reference page

勉強のために使用したサイト（ありがとうございます）

- [TensorRT | NVIDIA NGC](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt)
- [Container Release Notes :: NVIDIA Deep Learning TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/rel_21-06.html#rel_21-06)
- [Accelerating Inference In TF-TRT User Guide :: NVIDIA Deep Learning Frameworks Documentation](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html)
- [エッジAI(Jetson Nano 2GB)でTensorRTを用いて推論高速化をしてみました | GMOインターネットグループ 次世代システム研究室](https://recruit.gmo.jp/engineer/jisedai/blog/jetson_nano_tensorrt/)
- [ksriyudthsak/jetson_nano_study](https://github.com/ksriyudthsak/jetson_nano_study)
- [TensorFlowの画像識別モデルをTensorFlow-TensorRTで推論高速化 - Qiita](https://qiita.com/m_ogushi/items/fb9538ba24b81b57574a)
- [TensorFlow 2.X & TensorRT による推論高速化の第一歩 - Qiita](https://qiita.com/miya_sh/items/57356a8522c5499e890e)
- [JetPack SDK | NVIDIA Developer](https://developer.nvidia.com/embedded/jetpack)

- [tensorflow/tensorrt: TensorFlow/TensorRT integration](https://github.com/tensorflow/tensorrt/tree/master/tftrt/examples/image_classification)
- [Leveraging TensorFlow-TensorRT integration for Low latency Inference — The TensorFlow Blog](https://blog.tensorflow.org/2021/01/leveraging-tensorflow-tensorrt-integration.html)
