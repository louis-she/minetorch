<template>
  <div class="experiment-dataset">
    <el-breadcrumb separator-class="el-icon-arrow-right">
      <el-breadcrumb-item :to="{ path: '/' }">Experiments</el-breadcrumb-item>
      <el-breadcrumb-item>{{ experimentName }}</el-breadcrumb-item>
    </el-breadcrumb>

    <el-steps :space="200" :active="0" simple>
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
import dateFormat from 'dateformat'
import moment from 'moment'

export default {
  components: {
  },
  data () {
    return {
      experimentName: null
    }
  },
  mounted () {
  },
  methods: {
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
</style>
