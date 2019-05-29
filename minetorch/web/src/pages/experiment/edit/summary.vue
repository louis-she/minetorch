<template>
  <div class="experiment-summary">
    <div>
      <el-card class="box-card main">
        <div class="title">
          <h1>🚀ALL SET🚀</h1>
          <p>The training process now can be started at <strong>anytime</strong>, <strong>anywhere</strong> you want. It's time to grab a coffee!</p>
        </div>
        <div class="buttons">
          <el-tooltip class="item" effect="dark" placement="top"
                      content="Will train the model on current machine(which start up this Minetorch Server)">
            <el-button type="success" size="small" @click="handleTrainingButtonClicked">Start Training</el-button>
          </el-tooltip>
          <h3>Or, if you like to train it on another machine</h3>
          <el-tooltip class="item" effect="dark" placement="top"
                      content="Train the model anywhere you like, but you can still monitor the training process here">
            <el-button type="primary" size="small">Train with Docker</el-button>
          </el-tooltip>
        </div>
      </el-card>
    </div>
    <el-dialog :visible.sync="termDialogVisible" :close-on-click-modal="false" title="New Experiment" width="820px" @closed="closeTermDialog">
      <log attach="http://127.0.0.1:8000/server_log" />
    </el-dialog>
  </div>
</template>
<script>
import io from 'socket.io-client'
import log from 'components/log-com'

export default {
  components: {
    log
  },
  data () {
    return {
      termDialogVisible: false
    }
  },
  computed: {
    publishUrl() {
      return `/api/experiments/${this.experimentId}/publish`
    },
    trainingUrl() {
      return `/api/experiments/${this.experimentId}/training`
    }
  },
  mounted () {
    this.experimentId = this.$route.params.experimentId
  },
  methods: {
    async handleTrainingButtonClicked() {
      await this.ajax.post(this.publishUrl)
      await this.ajax.post(this.trainingUrl)
      this.termDialogVisible = true
    },
    closeTermDialog() {
      this.termDialogVisible = false
    }
  }
}
</script>
<style lang="scss">
.experiment-summary {
  .title {
    text-align: center;
  }

  .buttons {
    margin-top: 40px;
    text-align: center;
  }

  .el-card__body {
    width: 533px;
    margin: 0 auto;
  }
}
</style>
