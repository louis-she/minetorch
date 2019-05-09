<template>
  <el-form-item :label="label">
    <el-input v-model="formData[name]" :placeholder="help" />
  </el-form-item>
</template>
<script>
import formFieldMixin from 'components/form-fields/form-field'

export default {
  components: {
  },
  mixins: [formFieldMixin],
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
