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
            <el-button type="success" size="small">Start Training</el-button>
          </el-tooltip>
          <h3>Or, if you like to train it on another machine</h3>
          <el-tooltip class="item" effect="dark" placement="top"
                      content="Train the model anywhere you like, but you can still monitor the training process here">
            <el-button type="primary" size="small">Train with Docker</el-button>
          </el-tooltip>
        </div>
      </el-card>
    </div>
  </div>
</template>
<script>
export default {
  data () {
    return {
    }
  },
  computed: {
  },
  mounted () {
  },
  methods: {
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
