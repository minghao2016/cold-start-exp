import org.grouplens.lenskit.ItemScorer
import org.grouplens.lenskit.eval.data.CSVDataSourceBuilder
import org.grouplens.lenskit.eval.data.GenericDataSource
import org.grouplens.lenskit.eval.data.crossfold.RandomOrder
import org.grouplens.lenskit.eval.data.traintest.GenericTTDataBuilder
import org.grouplens.lenskit.eval.data.traintest.TTDataSet
import org.grouplens.lenskit.eval.metrics.predict.CoveragePredictMetric
import org.grouplens.lenskit.eval.metrics.predict.MAEPredictMetric
import org.grouplens.lenskit.eval.metrics.predict.NDCGPredictMetric
import org.grouplens.lenskit.eval.metrics.predict.RMSEPredictMetric
import org.grouplens.lenskit.iterative.IterationCount
import org.grouplens.lenskit.knn.NeighborhoodSize
import org.grouplens.lenskit.knn.item.*
import org.grouplens.lenskit.knn.item.model.ItemItemModel
import org.grouplens.lenskit.knn.user.*
import org.grouplens.lenskit.baseline.*
import org.grouplens.lenskit.mf.funksvd.FeatureCount
import org.grouplens.lenskit.mf.funksvd.FunkSVDItemScorer
import org.grouplens.lenskit.transform.normalize.*

import org.grouplens.ExtendedItemUserMeanScorer
import org.grouplens.lenskit.vectors.similarity.CosineVectorSimilarity
import org.grouplens.lenskit.vectors.similarity.VectorSimilarity

import java.util.zip.GZIPOutputStream

def zipFile = "${config.dataDir}/ml100k.zip"
def dataDir = config.get('mldata.directory', "${config.dataDir}/ml100k")

List<TTDataSet> coldStartTTData(String name) {
    int partition = 5
    List<TTDataSet> dataSets = new ArrayList<TTDataSet>(partition);
    File[] trainFiles = new File[partition];
    File[] testFiles = new File[partition];
    for (int i = 0; i < partition; i++) {
        trainFiles[i] = new File(String.format(
                "${config.dataDir}/ml100k-crossfold/train.%d." + name + ".csv", i));
        testFiles[i] = new File(String.format(
                "${config.dataDir}/ml100k-crossfold/test.%d." + name + ".csv", i));
    }
    for (int i = 0; i < partition; i++) {
        CSVDataSourceBuilder trainBuilder = new CSVDataSourceBuilder()
                .setFile(trainFiles[i]);
        CSVDataSourceBuilder testBuilder = new CSVDataSourceBuilder()
                .setFile(testFiles[i]);
        GenericTTDataBuilder ttBuilder = new GenericTTDataBuilder("coldstart." + name + "." + i);

        dataSets.add(ttBuilder.setTest(testBuilder.build())
                .setTrain(trainBuilder.build())
                .setAttribute("DataSet", "coldstart." + name)
                .setAttribute("Partition", i)
                .build());
    }
    return dataSets;
}
// This target unpacks the data
target('download') {
    perform {
        logger.warn("This analysis makes use of the MovieLens 100K data " +
                "set from GroupLens Research. Use of this data set is restricted to " +
                "non-commercial purposes and is only permitted in accordance with the " +
                "license terms.  More information is available at " +
                "<http://lenskit.grouplens.org/ML100K>.")
    }
    ant.mkdir(dir: config.dataDir)
    ant.get(src: 'http://www.grouplens.org/system/files/ml-100k.zip',
            dest: zipFile,
            skipExisting: true)
    ant.unzip(src: zipFile, dest: dataDir) {
        patternset {
            include name: 'ml-100k/*'
        }
        mapper type: 'flatten'
    }
}

def ml100k_source = csvfile("${dataDir}/u.data") {
    delimiter "\t"
    domain {
        minimum 1.0
        maximum 5.0
        precision 1.0
    }
}

def ml100k = target('crossfold') {
    requires 'download'

    crossfold {
        source ml100k_source
        test "${config.dataDir}/ml100k-crossfold/test.%d.csv"
        train "${config.dataDir}/ml100k-crossfold/train.%d.csv"
        order RandomOrder
        holdout 10
        partitions 5
    }
}

def coldstart = target('cold-start') {
    requires ml100k

    ant.exec(executable: 'python', dir: config.dataDir) {
        arg value: "${config.scriptDir}/cold_start.py"
        arg value: "${config.dataDir}/ml100k-crossfold/train.*.csv"
        arg value: "${config.dataDir}/ml100k-crossfold/test.*.csv"
    }
}

def baseline = algorithm("baseline") {
    bind ItemScorer to ItemMeanRatingItemScorer
    set MeanDamping to 5.0d
}

// Let's define some algorithms
def iiBase = {
    // use the item-item rating predictor with a baseline and normalizer
    bind ItemScorer to ItemItemScorer
    bind VectorSimilarity to CosineVectorSimilarity
    bind(BaselineScorer, ItemScorer) to UserMeanItemScorer
    bind(UserMeanBaseline, ItemScorer) to ItemMeanRatingItemScorer
    bind UserVectorNormalizer to BaselineSubtractingUserVectorNormalizer

    // retain 500 neighbors in the model, use 30 for prediction
    set NeighborhoodSize to 30

}

def itemitemSim = algorithm("itemitemSim") {
    include iiBase
    set ModelSize to 0
}

def itemitem = algorithm("itemitem") {
    include iiBase
    set ModelSize to 500
}

def funksvd = algorithm("funksvd") {
    bind ItemScorer to FunkSVDItemScorer
    bind(BaselineScorer, ItemScorer) to UserMeanItemScorer
    bind(UserMeanBaseline, ItemScorer) to ItemMeanRatingItemScorer
    set MeanDamping to 5.0d
    set FeatureCount to 25
    set IterationCount to 125
}


void dumpModel(ItemItemModel model, String fn) {
    File file = new File(config.analysisDir, fn)
    file.withWriter { out ->
        for (item in model.itemUniverse) {
            for (entry in model.getNeighbors(item)) {
                long oitem = entry.getId();
                if (oitem >= item) {
                    out.println("${item},${oitem},${entry.getScore()}");
                }
            }
        }
    }

}

target('dump_sim') {
    requires 'download'
    File dir = new File(config.analysisDir)
    dir.mkdirs()
    File file = new File(config.analysisDir, "item_sim.csv")
    if (!file.exists()) {
        trainModel("dump-sim") {
            input ml100k_source
            algorithm itemitemSim
            action {
                dumpModel(it.get(ItemItemModel), "item_sim.csv");
            }
        }
    }
}

target('evaluate') {
    // this requires the ml100k target to be run first
    // can either reference a target by object or by name (as above)
    requires coldstart

    trainTest {
        // and just use the target as the data set. The evaluator will do the right thing.
        dataset coldStartTTData("popular")
        dataset coldStartTTData("entropy")

        // Three different types of output for analysis.
        output "${config.analysisDir}/eval-results.csv"
        predictOutput "${config.analysisDir}/eval-preds.csv"
        userOutput "${config.analysisDir}/eval-user.csv"
        recommendOutput "${config.analysisDir}/eval-topN-recommend.csv"

        metric CoveragePredictMetric
        metric RMSEPredictMetric
        metric NDCGPredictMetric
        metric MAEPredictMetric

        algorithm itemitem
        algorithm funksvd
    }
}

// After running the evaluation, let's analyze the results
target('analyze') {
    requires 'evaluate'
    // Run R. Note that the script is run in the analysis directory; you might want to
    // copy all R scripts there instead of running them from the source dir.
    ant.exec(executable: config["rscript.executable"], dir: config.analysisDir) {
        arg value: "${config.scriptDir}/chart.R"
    }
}

// By default, run the analyze target
defaultTarget 'analyze'