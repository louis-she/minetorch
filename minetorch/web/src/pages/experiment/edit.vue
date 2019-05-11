<template>
  <div class="experiment-dataset">
    <el-breadcrumb separator-class="el-icon-arrow-right">
      <el-breadcrumb-item :to="{ path: '/' }">Experiments</el-breadcrumb-item>
      <el-breadcrumb-item>{{ experimentName }}</el-breadcrumb-item>
    </el-breadcrumb>

    <el-steps :space="200" :active="activateStep" simple>
      <el-step title="Dataset" icon="el-icon-coin" />
      <el-step title="Dataflow" icon="el-icon-setting" />
      <el-step title="Model" icon="el-icon-cpu" />
      <el-step title="Optimizer" icon="el-icon-magic-stick" />
      <el-step title="Loss" icon="el-icon-sugar" />
    </el-steps>
    <router-view />
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
  watch: {
    '$route' (to, from) {
      this.activateStep = this.steps.indexOf(this.$route.params.componentName)
      this.experimentName = this.$route.params.componentName || 'datasets'
      this.experimentId = this.$route.params.experimentId
    }
  },
  mounted () {
    this.activateStep = this.steps.indexOf(this.$route.params.componentName)
    this.experimentName = this.$route.params.componentName || 'datasets'
    this.experimentId = this.$route.params.experimentId
  }
}
</script>
<style lang="scss">
.el-steps {
  margin-bottom: 30px;
}

.el-breadcrumb {
  margin-bottom: 30px;
}

.el-step__head.is-finish, .el-step__title.is-finish {
  color: #67C23A;
}
</style>
