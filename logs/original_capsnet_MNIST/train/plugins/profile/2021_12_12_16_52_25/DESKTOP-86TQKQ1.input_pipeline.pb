  *	p=
ף�p@2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch]��k�?!Lpx��E4@)]��k�?1Lpx��E4@:Preprocessing2�
LIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2�`TR'��?!��HM�2@)�`TR'��?1��HM�2@:Preprocessing2�
yIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2::ParallelMapV2::ParallelMapV2::ParallelMapV2R,���?!rY��1@)R,���?1rY��1@:Preprocessing2�
[Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2::ParallelMapV2��m��?!?�i�.@)��m��?1?�i�.@:Preprocessing2�
jIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2::ParallelMapV2::ParallelMapV2T㥛� �?!�mF�'@)T㥛� �?1�mF�'@:Preprocessing2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2::ParallelMapV2::ParallelMapV2::ParallelMapV2::Shuffle�7�0��?!�,WY:�@)�7�0��?1�,WY:�@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch�$]3�f�?!��/��B@)�$]3�f�?1��/��B@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism�~�T�?!����N"@)��_vO�?1�
�oyZ@:Preprocessing2F
Iterator::Model-σ��v�?!8��4N$@)8�*5{�e?1B�C���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.