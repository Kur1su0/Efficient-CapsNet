  *	�"���
�@2�
LIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2 7o��M@!����r�>@)7o��M@1����r�>@:Preprocessing2�
[Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2::ParallelMapV2 �S����@!O�	��<@)�S����@1O�	��<@:Preprocessing2�
yIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2::ParallelMapV2::ParallelMapV2::ParallelMapV2 wMHk*@!4���t;@)wMHk*@14���t;@:Preprocessing2�
jIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2::ParallelMapV2::ParallelMapV2 Кiq@!�t[�N�(@)Кiq@1�t[�N�(@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch+O ���?!�7�K���?)+O ���?1�7�K���?:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchN�»\ħ?!��V�?)N�»\ħ?1��V�?:Preprocessing2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2::ParallelMapV2::ParallelMapV2::ParallelMapV2::Shuffle *�-9(�?!7�mt��?)*�-9(�?17�mt��?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism���.4ױ?!+�T*#��?)�c�Cԗ?1!JK��?:Preprocessing2F
Iterator::Modelt$���~�?!�.mX���?)g���uz?1��p1��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.