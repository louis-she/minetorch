<template>
  <div class="experiment-dataset">
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
    }
  },
  mounted () {
  },
  methods: {
  }
}
</script>
<style lang="scss">
</style>
