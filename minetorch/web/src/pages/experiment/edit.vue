<template>
  <div class="experiment-dataset">
    <el-breadcrumb separator-class="el-icon-arrow-right">
      <el-breadcrumb-item :to="{ path: '/' }">Experiments</el-breadcrumb-item>
      <el-breadcrumb-item>{{ experimentName }}</el-breadcrumb-item>
    </el-breadcrumb>

    <el-steps :space="200" :active="activateStep" simple>
      <el-step title="Dataset" icon="el-icon-coin" @click.native="handleStepClick(0)"/>
      <el-step title="Dataflow" icon="el-icon-setting" @click.native="handleStepClick(1)"/>
      <el-step title="Model" icon="el-icon-cpu" @click.native="handleStepClick(2)"/>
      <el-step title="Optimizer" icon="el-icon-magic-stick" @click.native="handleStepClick(3)"/>
      <el-step title="Loss" icon="el-icon-sugar" @click.native="handleStepClick(4)"/>
      <el-step title="Coffee" icon="el-icon-coffee-cup" @click.native="toSummarizePage"/>
    </el-steps>
    <router-view :key="activateStep" />
  </div>
</template>
<script>
export default {
  data () {
    return {
      experimentName: '',
      activateStep: 0
    }
  },
  computed: {
    componentName() {
      return this.$route.params.componentName || 'summary'
    }
  },
  watch: {
    '$route' (to, from) {
      this.activateStep = this.steps.indexOf(this.componentName)
      console.log(this.activateStep)
      this.experimentName = this.componentName
      this.experimentId = this.$route.params.experimentId
    }
  },
  mounted () {
    this.activateStep = this.steps.indexOf(this.componentName)
    this.experimentName = this.componentName
    this.experimentId = this.$route.params.experimentId
  },
  methods: {
    toSummarizePage() {
      this.$router.push({
        name: 'EditExperimentSummary',
        params: {
          experimentId: this.experimentId,
          componentName: 'summary'
        }
      })
    },
    handleStepClick(step) {
      this.$router.push({
        name: 'EditExperimentComponent',
        params: {
          experimentId: this.experimentId,
          componentName: this.steps[step]
        }
      })
    }
  }
}
</script>
<style lang="scss">
.experiment-dataset {
  .el-step__title, .el-step__head {
    cursor: pointer;
  }
  .el-steps {
    margin-bottom: 30px;
  }

  .el-breadcrumb {
    margin-bottom: 30px;
  }

  .el-step__head.is-finish, .el-step__title.is-finish {
    color: #67C23A;
  }
}
</style>
