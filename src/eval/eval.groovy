import org.grouplens.lenskit.ItemScorer
import org.grouplens.lenskit.core.Transient
import org.grouplens.lenskit.data.dao.EventDAO
import org.grouplens.lenskit.data.dao.UserDAO
import org.grouplens.lenskit.eval.ExecutionInfo
import org.grouplens.lenskit.eval.data.CSVDataSourceBuilder
import org.grouplens.lenskit.eval.data.crossfold.RandomOrder
import org.grouplens.lenskit.eval.data.traintest.GenericTTDataBuilder
import org.grouplens.lenskit.eval.data.traintest.QueryData
import org.grouplens.lenskit.eval.data.traintest.TTDataSet
import org.grouplens.lenskit.eval.metrics.predict.CoveragePredictMetric
import org.grouplens.lenskit.eval.metrics.predict.MAEPredictMetric
import org.grouplens.lenskit.eval.metrics.predict.NDCGPredictMetric
import org.grouplens.lenskit.eval.metrics.predict.RMSEPredictMetric
import org.grouplens.lenskit.eval.metrics.topn.ItemSelectors
import org.grouplens.lenskit.external.ExternalProcessItemScorerBuilder
import org.grouplens.lenskit.iterative.IterationCount
import org.grouplens.lenskit.knn.MinNeighbors
import org.grouplens.lenskit.knn.NeighborhoodSize
import org.grouplens.lenskit.knn.item.*
import org.grouplens.lenskit.knn.item.model.ItemItemModel
import org.grouplens.lenskit.baseline.*
import org.grouplens.lenskit.mf.funksvd.FeatureCount
import org.grouplens.lenskit.mf.funksvd.FunkSVDItemScorer
import org.grouplens.lenskit.transform.normalize.*
import org.grouplens.lenskit.eval.metrics.topn.*
import org.grouplens.lenskit.transform.threshold.ThresholdValue
import org.grouplens.lenskit.vectors.similarity.CosineVectorSimilarity
import org.grouplens.lenskit.vectors.similarity.VectorSimilarity
import org.hamcrest.Matchers
import javax.inject.Inject
import javax.inject.Provider



def zipFile = "${config.dataDir}/ml-1m.zip"
def dataDir = config.get('mldata.directory', "${config.dataDir}/ml-1m")
def recommendFile = "${config.analysisDir}/recommend-model.csv"

class ExternalItemMeanScorerBuilder implements Provider<ItemScorer> {
    ExecutionInfo exInfo
    String testFile
    String trainFile
    String clusterName
    String predictionName
    String partition
    String scoreType

    @Inject
    public ExternalItemMeanScorerBuilder(ExecutionInfo info) {
        exInfo = info
        trainFile = String.format("/Users/shuochang/Dropbox/cold_start/cold-start-exp/target/data/ml-recent/train.%d.csv",
                info.dataAttributes['Partition'])
        testFile = String.format("/Users/shuochang/Dropbox/cold_start/cold-start-exp/target/data/ml-recent/test.%d.csv",
                info.dataAttributes['Partition'])
        clusterName = info.algoAttributes['ClusterName']
        predictionName = info.algoAttributes['PredictionName']
        partition = info.dataAttributes['Partition']
        scoreType = info.algoAttributes['ScoreType']
    }

    @Override
    ItemScorer get() {
        def wrk = new File("external-scratch")
        wrk.mkdirs()
        def builder = new ExternalProcessItemScorerBuilder()
        // Note: don't use file names because it will interact badly with crossfolding
        return builder.setWorkingDir(wrk)
                .setExecutable("python")
                .addArgument("/Users/shuochang/Dropbox/cold_start/cold-start-exp/src/eval/cluster_recommender.py")
                .addArgument(trainFile)
                .addArgument(testFile)
                .addArgument("/Users/shuochang/Dropbox/cold_start/cold-start-exp/target/analysis/recommend-model.csv")
                .addArgument(clusterName)
                .addArgument(predictionName)
                .addArgument(partition)
                .addArgument(scoreType)
                .build()
    }
}

List<TTDataSet> customTTData(String name) {
    int partition = 5
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
        GenericTTDataBuilder ttBuilder = new GenericTTDataBuilder(name + "." + i);

        dataSets.add(ttBuilder.setTest(testBuilder.build())
                .setTrain(trainBuilder.build())
                .setAttribute("DataSet", name)
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

def cv_recent = target('crossfold') {
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
    requires "crossfold"

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
        arg value: "4"
        arg value: "8"
        arg value: "12"
        arg value: "16"
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
    set FeatureCount to 50
    set IterationCount to 125
}

def external = algorithm("external") {
    attributes['cluster_name'] = 'bias_spectral_10'
    attributes['prediction_name'] = 'itemitem'
    bind ItemScorer toProvider ExternalItemMeanScorerBuilder
}

def data_names = ['bias_spectral_4', 'bias_spectral_8', 'bias_spectral_12', 'bias_spectral_16',
                  'original_spectral_4', 'original_spectral_8', 'original_spectral_12', 'original_spectral_16']


target('cluster-recommend'){
//    requires 'cluster'

    trainTest{
        componentCacheDirectory "${config.analysisDir}/cache"
        for (dn in data_names) {
            dataset customTTData(dn)
        }

        recommendOutput recommendFile

        metric RMSEPredictMetric

        algorithm itemitem
    }
}

//def cluster_names = ['bias_spectral_5', 'bias_spectral_10', 'bias_spectral_15',
//                     'bias_spectral_svd_5', 'bias_spectral_svd_10', 'bias_spectral_svd_15',
//                     'original_spectral_5', 'original_spectral_10', 'original_spectral_15',
//                     'original_spectral_svd_5', 'original_spectral_svd_10', 'original_spectral_svd_15']
def cluster_names = ['bias_spectral_10', 'original_spectral_10']
def prediction_names = ['itemitem']
def score_type = ['optimal', 'simulation']

target('cluster-evaluate') {
    // this requires the ml100k target to be run first
    // can either reference a target by object or by name (as above)
    requires 'cluster-recommend'
    requires 'crossfold'
    trainTest {

        dataset cv_recent

        componentCacheDirectory "${config.analysisDir}/cache"


        // Three different types of output for analysis.
//        output "${config.analysisDir}/eval-results-topn-fake.csv"
//        predictOutput "${config.analysisDir}/eval-preds-topn-fake.csv"
//        userOutput "${config.analysisDir}/eval-user-topn-fake.csv"
//        recommendOutput "${config.analysisDir}/eval-recommend-fake.csv"
        output "${config.analysisDir}/external-results-change-k.csv"
        predictOutput "${config.analysisDir}/external-preds-change-k.csv"
        userOutput "${config.analysisDir}/external-user-preds-change-k.csv"


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

        for (cn in cluster_names){
            for (pn in prediction_names){
                for (st in score_type){
                    algorithm("external") {
                        attributes['ClusterName'] = cn
                        attributes['PredictionName'] = pn
                        attributes['ScoreType'] = st
                        bind ItemScorer toProvider ExternalItemMeanScorerBuilder
//                        bind ItemScorer to FallbackItemScorer
//                        bind (PrimaryScorer, ItemScorer) toProvider ExternalItemMeanScorerBuilder
//                        bind (BaselineScorer, ItemScorer) to ItemMeanRatingItemScorer
                    }
                }
            }
        }
    }
}


target('baseline-evaluate') {

//    requires 'coldstart'
    trainTest {

        componentCacheDirectory "${config.analysisDir}/cache"
        dataset customTTData('popular_5')
        dataset customTTData('popular_10')
        dataset customTTData('popular_15')
        dataset customTTData('entropy_zero_5')
        dataset customTTData('entropy_zero_10')
        dataset customTTData('entropy_zero_15')
        // Three different types of output for analysis.
//        output "${config.analysisDir}/eval-results-topn-fake.csv"
//        predictOutput "${config.analysisDir}/eval-preds-topn-fake.csv"
//        userOutput "${config.analysisDir}/eval-user-topn-fake.csv"
//        recommendOutput "${config.analysisDir}/eval-recommend-fake.csv"
        output "${config.analysisDir}/baseline-results.csv"
        predictOutput "${config.analysisDir}/baseline-preds.csv"
        userOutput "${config.analysisDir}/baseline-user-preds.csv"


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


//
//void dumpModel(ItemItemModel model, String fn) {
//    File file = new File(config.analysisDir, fn)
//    file.withWriter { out ->
//        for (item in model.itemUniverse) {
//            for (entry in model.getNeighbors(item)) {
//                long oitem = entry.getId();
//                if (oitem >= item) {
//                    out.println("${item},${oitem},${entry.getScore()}");
//                }
//            }
//        }
//    }
//
//}
//target('dump_sim') {
//    requires 'download'
//    File dir = new File(config.analysisDir)
//    dir.mkdirs()
//    File file = new File(config.analysisDir, "item_sim.csv")
//    if (!file.exists()) {
//        trainModel("dump-sim") {
//            input ml1m_source
//            algorithm itemitemSim
//            action {
//                dumpModel(it.get(ItemItemModel), "item_sim.csv");
//            }
//        }
//    }
//}

// After running the evaluation, let's analyze the results
//target('analyze') {
//    requires 'evaluate'
//    // Run R. Note that the script is run in the analysis directory; you might want to
//    // copy all R scripts there instead of running them from the source dir.
//    ant.exec(executable: config["rscript.executable"], dir: config.analysisDir) {
//        arg value: "${config.scriptDir}/chart.R"
//    }
//}

// By default, run the analyze target
defaultTarget 'analyze'