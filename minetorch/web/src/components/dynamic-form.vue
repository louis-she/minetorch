<template>
  <div class="dynamic-form">
    <el-form ref="form" :model="componentSettings" label-width="120px">
      <component v-for="field in schema" :is="field.input" :key="field.label"
                 v-bind="field" :form-data="componentSettings" />
    </el-form>
  </div>
</template>
<script>

import string from './form-fields/string'

export default {
  components: {
    string
  },
  props: {
    schema: {
      type: Array,
      default: function () {
        return [ {label: 'haha', input: 'string', help: 'this is string', name: 'test'} ]
      }
    },
    submitUrl: {
      type: String,
      default: ''
    }
  },
  data () {
    return {
      componentSettings: {
        test: ''
      }
    }
  },
  mounted () {
  },
  methods: {
    debug() {
      console.log(this.componentSettings)
    }
  }
}
</script>
<style lang="scss">
.el-input {
  width: 300px;
}
</style>
