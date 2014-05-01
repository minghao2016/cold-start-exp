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
import org.grouplens.lenskit.eval.metrics.topn.ItemSelectors
import org.grouplens.lenskit.eval.metrics.topn.NDCGTopNMetric
import org.grouplens.lenskit.iterative.IterationCount
import org.grouplens.lenskit.knn.MinNeighbors
import org.grouplens.lenskit.knn.NeighborhoodSize
import org.grouplens.lenskit.knn.item.*
import org.grouplens.lenskit.knn.item.model.ItemItemModel
import org.grouplens.lenskit.knn.user.*
import org.grouplens.lenskit.baseline.*
import org.grouplens.lenskit.mf.funksvd.FeatureCount
import org.grouplens.lenskit.mf.funksvd.FunkSVDItemScorer
import org.grouplens.lenskit.transform.normalize.*
import org.grouplens.lenskit.eval.metrics.topn.*


import org.grouplens.ExtendedItemUserMeanScorer
import org.grouplens.lenskit.transform.threshold.ThresholdValue
import org.grouplens.lenskit.vectors.similarity.CosineVectorSimilarity
import org.grouplens.lenskit.vectors.similarity.VectorSimilarity
import org.hamcrest.Description
import org.hamcrest.Matcher
import org.hamcrest.Matchers

import java.util.zip.GZIPOutputStream

def zipFile = "${config.dataDir}/ml-1m.zip"
def dataDir = config.get('mldata.directory', "${config.dataDir}/ml-1m")

List<TTDataSet> coldStartTTData(String name) {
    int partition = 1
    List<TTDataSet> dataSets = new ArrayList<TTDataSet>(partition);
    File[] trainFiles = new File[partition];
    File[] testFiles = new File[partition];
    for (int i = 0; i < partition; i++) {
        trainFiles[i] = new File(String.format(
                "${config.dataDir}/ml-recent/train.%d." + name + ".csv", i));
        testFiles[i] = new File(String.format(
                "${config.dataDir}/ml-recent/test.%d." + name + ".csv", i));
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
                "<http://lenskit.grouplens.org/ML1M>.")
    }
    ant.mkdir(dir: config.dataDir)
    ant.get(src: 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
            dest: zipFile,
            skipExisting: true)
    ant.unzip(src: zipFile, dest: dataDir) {
        patternset {
            include name: 'ml-1m/*'
        }
        mapper type: 'flatten'
    }
}

def ml1m_source = csvfile("${dataDir}/ratings.dat") {
    delimiter "::"
    domain {
        minimum 1.0
        maximum 5.0
        precision 0.5
    }
}

def ml1m_fited = csvfile("/Users/shuochang/Dropbox/cold_start/data/ml-10M100K/ratings_fited.dat") {
    delimiter ","
    domain{
        minimum 1.0
        maximum 5.0
        precision 0.5d
    }
}

def ml_fited = csvfile("/Users/shuochang/Dropbox/cold_start/data/movielens/recent_rating.csv") {
    delimiter ","
    domain{
        minimum 1.0
        maximum 5.0
        precision 0.5d
    }
}

def ml1m = target('crossfold') {
//    requires 'download'

    crossfold {
        source ml_fited
        test "${config.dataDir}/ml-recent/test.%d.csv"
        train "${config.dataDir}/ml-recent/train.%d.csv"
        order RandomOrder
        holdoutFraction 0.2d
        partitions 5
    }
}

def coldstart = target('cold-start') {
    //requires ml1m

    ant.exec(executable: 'python', dir: config.dataDir) {
        arg value: "${config.scriptDir}/cold_start.py"
        arg value: "${config.dataDir}/ml-recent/train.[0-9].csv"
        arg value: "${config.dataDir}/ml-recent/test.[0-9].csv"
    }
}

def cluster = target('cluster'){
    ant.exec(executable: 'python', dir: config.dataDir) {
        arg value: "${config.scriptDir}/cluster_rating_tt.py"
        arg value: "${config.dataDir}/ml-recent/train.[0-9].csv"
        arg value: "${config.dataDir}/ml-recent/test.[0-9].csv"
        arg value: "${config.dataDir}/tag_genome.pkl"
        arg value: "${config.dataDir}/movies.pkl"
        arg value: "--algorithm"
        arg value: "spectral"
        arg value: "spectral_svd"
        arg value: "--k"
        arg value: "5"
        arg value: "10"
        arg value: "15"
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

}

def itemitemSim = algorithm("itemitemSim") {
    include iiBase
    set ModelSize to 0
}

def itemitem = algorithm("itemitem") {
    include iiBase
    set MeanDamping to 5.0d
    set MinNeighbors to 2
    set ThresholdValue to 0.1
    set ModelSize to 3000
    set NeighborhoodSize to 20
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
            input ml1m_source
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
    //requires coldstart

    trainTest {

        componentCacheDirectory "${config.analysisDir}/cache"
        // and just use the target as the data set. The evaluator will do the right thing.
//        dataset coldStartTTData("pseudo_rating")
//        dataset coldStartTTData("pseudo_tag")
//        dataset coldStartTTData("pseudo_p1")
//        dataset coldStartTTData("pseudo_p2")
//        dataset coldStartTTData("popular_5")
//        dataset coldStartTTData("sum_1_fake")
//        dataset coldStartTTData("sum_2_fake")
//        dataset coldStartTTData("sum_0_fake")
//        dataset coldStartTTData("sum_20_fake")

        dataset coldStartTTData("bias_10")
        dataset coldStartTTData("orig_10")
//        dataset coldStartTTData("popular_5")
//        dataset coldStartTTData("popular_10")
//        dataset coldStartTTData("popular_15")
//        dataset coldStartTTData("entropy_zero_5")
//        dataset coldStartTTData("entropy_zero_10")
//        dataset coldStartTTData("entropy_zero_15")


        // Three different types of output for analysis.
//        output "${config.analysisDir}/eval-results-topn-fake.csv"
//        predictOutput "${config.analysisDir}/eval-preds-topn-fake.csv"
//        userOutput "${config.analysisDir}/eval-user-topn-fake.csv"
//        recommendOutput "${config.analysisDir}/eval-recommend-fake.csv"
//        output "${config.analysisDir}/eval-recent-results.csv"
//        predictOutput "${config.analysisDir}/eval-recent-preds.csv"
//        userOutput "${config.analysisDir}/eval-recent-user-preds.csv"
        recommendOutput "${config.analysisDir}/eval-recent-recommend.csv"


        metric CoveragePredictMetric
        metric RMSEPredictMetric
        metric NDCGPredictMetric
        metric MAEPredictMetric
        def topNConfig = {
            listSize 30
            candidates ItemSelectors.addNRandom(ItemSelectors.testItems(), 1000)
//            exclude ItemSelectors.setDifference(ItemSelectors.testItems(), ItemSelectors.testItems())
        }

        metric(topNnDCG(topNConfig))
        metric(topNPopularity(topNConfig))
        metric new PrecisionRecallTopNMetric.Builder().setListSize(30)
                    .setCandidates(ItemSelectors.addNRandom(ItemSelectors.testItems(), 1000))
//                    .setExclude(ItemSelectors.setDifference(ItemSelectors.testItems(), ItemSelectors.testItems()))
                    .setGoodItems(ItemSelectors.testRatingMatches(Matchers.greaterThanOrEqualTo(4.0d)))
                    .build()

        algorithm itemitem
        algorithm funksvd
        algorithm baseline
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