  *	�(\��]k@2�
jIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2::ParallelMapV2::ParallelMapV2�M(D�!�?!��q���1@)�M(D�!�?1��q���1@:Preprocessing2�
LIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2���jHܣ?!/8�Sη1@)���jHܣ?1/8�Sη1@:Preprocessing2�
[Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2::ParallelMapV2j���v��?!�!	vi1@)j���v��?1�!	vi1@:Preprocessing2�
yIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2::ParallelMapV2::ParallelMapV2::ParallelMapV23P�>�?!��V�|+1@)3P�>�?1��V�|+1@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch^���T��?!��
H�"@)^���T��?1��
H�"@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatchj>"�D�?!�|w%L @)j>"�D�?1�|w%L @:Preprocessing2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2::ParallelMapV2::ParallelMapV2::ParallelMapV2::Shuffle����a�?!vϊm@)����a�?1vϊm@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismG����g�?!���i�,@)��x@ل?1?�C|�@:Preprocessing2F
Iterator::Modelz6�>W[�?!yv�.@)g���uj?1�W�a��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.